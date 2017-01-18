# Adapted from implementation by 
# Parag K. Mital, Jan 2016


#=======================================================================================
# Import statements

# Libraries
import numpy as np
import math
import tensorflow as tf
import matplotlib.pyplot as plt

# Local modules
from libs.activations import lrelu
from libs.utils import corrupt
from utils.utils import format_as_percent
from utils.utils import progress_bar

# Config info
from configparser import ConfigParser
config = ConfigParser()
config.read('config.ini')
resized_root = config.get('main', 'resized_root')


#=======================================================================================
# Hyperparameters

IMAGE_DIM = 256
BATCH_SIZE = 5
MIN_AFTER_DEQUEUE = 1000
N_EPOCHS = 3000
LEARNING_RATE = 0.01
LATENT_DIM = 64**2
FILE_NAME = resized_root + '/' + str(IMAGE_DIM) + '_images_and_names.tfrecords'


#=======================================================================================
# Image loading

def read_and_decode_input_pair(file_name, image_dim):
    file_name_queue = tf.train.string_input_producer([file_name], num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_pair = reader.read(file_name_queue)
    components = tf.parse_single_example(serialized_pair,
                                         features={'label_matrix': tf.FixedLenFeature([20*39], tf.float32),
                                                   'image': tf.FixedLenFeature([image_dim**2], tf.float32)})
    image = components['image']
    image = tf.div(image, 255)
    label_matrix = components['label_matrix']
    return image, label_matrix

image, label_matrix = read_and_decode_input_pair(FILE_NAME, IMAGE_DIM)
images_batch, labels_batch = tf.train.shuffle_batch(
    [image, label_matrix],
    batch_size=BATCH_SIZE,
    capacity=(3 * MIN_AFTER_DEQUEUE + BATCH_SIZE),
    min_after_dequeue=MIN_AFTER_DEQUEUE
)


#=======================================================================================
# Graph construction

class autoencoder(object):
    def __init__(self):
        self.input_shape = [None, IMAGE_DIM ** 2]
        self.latent_dim = LATENT_DIM
        self.layers_deep = 3
        self.corruption = False

        # self.n_filters = [1, 10, 10] + [10, 10, 10]*4 + [3, 3, 3]
        # self.striders = [2, 1, 1] * 6
        # self.filter_sizes = [7, 3, 3] + [3, 3, 3]*5

        # self.duplication = 2
        # self.n_filters = [1, 10] + [10, 10]*4 + [3, 3]
        # self.striders = [2, 1] * 6
        # self.filter_sizes = [7, 3] + [3, 3]*5

        self.duplication = 1
        self.n_filters = [1] + [10]*4 + [3]
        self.striders = [2] * 6
        self.filter_sizes = [7] + [3]*5

        self.initialize_and_validate_x()
        self.build_graph()

    def set_layers(self, depth):
        self.layers_deep = depth

    def initialize_and_validate_x(self):
        # Input to the network
        self.x = tf.placeholder(
            tf.float32, self.input_shape, name='self.x')

        # Ensure 2-d is converted to square tensor.
        if len(self.x.get_shape()) == 2:
            x_dim = np.sqrt(self.x.get_shape().as_list()[1])
            if x_dim != int(x_dim):
                raise ValueError('Unsupported input dimensions')
            x_dim = int(x_dim)
            self.x_tensor = tf.reshape(
                self.x, [-1, x_dim, x_dim, self.n_filters[0]])
        elif len(self.x.get_shape()) == 4:
            self.x_tensor = self.x
        else:
            raise ValueError('Unsupported input dimensions')


    def init_weight(self, shape):
        # init = tf.random_uniform(sizes,
        #                          -1.0 / math.sqrt(n_input),
        #                          1.0 / math.sqrt(n_input))
        init = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(init)

    def init_bias(self, shape):
        return tf.Variable(tf.zeros(shape))


    def fully_connected(self):
        pre_latent_shape = self.current_input.get_shape().as_list()[1:]
        pre_latent_size = np.product(pre_latent_shape)
        self.compression = tf.divide(float(self.latent_dim), pre_latent_size)

        W_fc_in = self.init_weight([pre_latent_size, self.latent_dim])
        b_fc_in = self.init_bias(self.latent_dim)
        W_fc_out = self.init_weight([self.latent_dim, pre_latent_size])
        b_fc_out = self.init_bias(pre_latent_size)
        self.current_input = tf.reshape(self.current_input, [-1, pre_latent_size])
        self.z = lrelu(tf.matmul(self.current_input, W_fc_in) + b_fc_in)
        self.current_input = lrelu(tf.matmul(self.z, W_fc_out) + b_fc_out)
        self.current_input = tf.reshape(self.current_input, [-1] + pre_latent_shape)


    def encoder_step(self, n_input, n_output, filter_size, stride_step):
        self.shapes.append(self.current_input.get_shape().as_list())
        bn_mean, bn_var = tf.nn.moments(self.current_input, [0, 1, 2, 3])
        normalized = tf.nn.batch_normalization(self.current_input, bn_mean, bn_var, offset=None, scale=None, variance_epsilon=1e-6)
        W_conv = self.init_weight([filter_size, filter_size, n_input, n_output])
        self.encoder.append(W_conv)
        b_conv = self.init_bias([n_output])
        convolved = tf.nn.conv2d(normalized, W_conv, strides=[1, stride_step, stride_step, 1], padding='SAME')
        output = lrelu(tf.add(convolved, b_conv))
        self.current_input = output


    def decoder_step(self, weight_shape, shape, stride_step):
        bn_mean, bn_var = tf.nn.moments(self.current_input, [0, 1, 2, 3])
        normalized = tf.nn.batch_normalization(self.current_input, bn_mean, bn_var, offset=None, scale=None, variance_epsilon=1e-6)
        W_deconv = self.init_weight(weight_shape)
        bias_shape = W_deconv.get_shape().as_list()[2]
        b_deconv = self.init_bias(bias_shape)
        deconvolved = tf.nn.conv2d_transpose(
                normalized, W_deconv,
                tf.pack([tf.shape(self.x)[0], shape[1], shape[2], shape[3]]),
                strides=[1, stride_step, stride_step, 1], padding='SAME')
        output = lrelu(tf.add(deconvolved, b_deconv))
        self.current_input = output


    def build_graph(self):
        self.current_input = self.x_tensor
        if self.corruption:
            self.current_input = corrupt(self.current_input) # Optionally apply denoising autoencoder

        self.encoder = []
        self.shapes = []
        occ = max(0, 5 - self.layers_deep) * self.duplication
        qz = 6 * self.duplication
        for layer_i in range(1, qz - occ):
            n_output = self.n_filters[layer_i]
            n_input = self.current_input.get_shape().as_list()[3]
            self.encoder_step(n_input, n_output, self.filter_sizes[layer_i - 1], self.striders[layer_i - 1])

        if self.layers_deep == 6:
            self.fully_connected()

        for layer_i in range(0, qz - 1 - occ):
            reversed_index = qz - occ - 2 - layer_i
            shape = self.shapes[reversed_index]
            weight_shape = self.encoder[reversed_index].get_shape()
            self.decoder_step(weight_shape, shape, self.striders[reversed_index])

        self.y = self.current_input  # Pass the current state from the previous operations
        self.cost = tf.reduce_sum(tf.square(self.y - self.x_tensor))


ae = autoencoder()
optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(ae.cost)

# We create a session to use the graph
tf.Graph().as_default()
sess = tf.Session()
coord = tf.train.Coordinator()
tf.train.start_queue_runners(sess=sess, coord=coord)
sess.run(tf.initialize_all_variables())

# Fit all training data
costs = []
# for depth in range(1,2):
#     ae.set_layers(depth)
depth = 1
for epoch_i in range(N_EPOCHS):
    for batch_i in range(1): #TODO
        batch_xs, _ = sess.run([images_batch, labels_batch])
        # batch_xs_f = np.reshape(batch_xs, (BATCH_SIZE, IMAGE_DIM, IMAGE_DIM))
        # batch_xs_f = [np.fft.fftshift(np.fft.fft2(batch_xs_f[i])) for i in range(len(batch_xs))]
        # batch_xs_f = np.reshape(batch_xs, (BATCH_SIZE, IMAGE_DIM * IMAGE_DIM))=
        train = batch_xs
        sess.run(optimizer, feed_dict={ae.x: train})
    if epoch_i % 25 == 0:
        format_epoch = str(epoch_i) #TODO
        cost_reported = sess.run(ae.cost, feed_dict={ae.x: train})
        format_cost = str(cost_reported) #TODO
        info_string = 'Depth: ' + str(depth) + ' | Epoch: ' + format_epoch + ' | Cost: ' + format_cost
        progress_bar(info_string, epoch_i, N_EPOCHS)  #TODO
        costs += [cost_reported]

# Plot example reconstructions
n_examples = 5
test_xs, _ = sess.run([images_batch, labels_batch])
# test_xs_f = np.reshape(test_xs, (BATCH_SIZE, IMAGE_DIM, IMAGE_DIM))
# test_xs_f = [np.fft.fftshift(np.fft.fft2(test_xs_f[i])) for i in range(len(test_xs_f))]
# test_xs_f = np.reshape(test_xs_f, (BATCH_SIZE, IMAGE_DIM * IMAGE_DIM))

test_xs_norm = test_xs #new
recon = sess.run(ae.y, feed_dict={ae.x: test_xs_norm})


#=======================================================================================
# Plot reconstructions

plt.figure(figsize=(8, 12))
for i in range(5):
    plt.subplot(5, 2, 2*i + 1)
    plt.imshow(test_xs[i].reshape(IMAGE_DIM, IMAGE_DIM), vmin=0, vmax=1, cmap="gray")
    plt.title("Test input")
    plt.colorbar()
    plt.subplot(5, 2, 2*i + 2)
    plt.imshow(recon[i].reshape(IMAGE_DIM, IMAGE_DIM), vmin=0, vmax=1, cmap="gray")
    plt.title("Reconstruction")
    plt.colorbar()
plt.tight_layout()
plt.show()


# plt.figure(figsize=(8, 12))
# for i in range(5):
#     plt.subplot(5, 2, 2*i + 1)
#     plt.imshow(np.fft.ifft2(np.fft.ifftshift(test_xs_f[i])).reshape(IMAGE_DIM, IMAGE_DIM), vmin=0, vmax=1, cmap="gray")
#     plt.title("Test input")
#     plt.colorbar()
#     plt.subplot(5, 2, 2*i + 2)
#     plt.imshow(np.fft.ifft2(np.fft.ifftshift(recon[i])).reshape(IMAGE_DIM, IMAGE_DIM), vmin=0, vmax=1, cmap="gray")
#     plt.title("Reconstruction")
#     plt.colorbar()
# plt.tight_layout()
# plt.show()


plt.plot(costs)
plt.xscale('log')
plt.yscale('log')
plt.show()


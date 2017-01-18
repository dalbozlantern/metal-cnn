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
N_EPOCHS = 1000
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

def init_weight(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)

def init_bias(shape):
    return tf.Variable(tf.zeros(shape))


class encoder(object):
    def __init__(self, input_data):
        self.bn_means = [None]
        self.bn_vars = [None]
        self.weights = [None]
        self.biases = [None]
        self.outputs = [input_data]
        self.counter = 0

    def step(self, n_output, filter_size, stride_step):
        self.counter += 1
        n_input = self.outputs[self.counter - 1].get_shape().as_list()[3]
        bn_mean, bn_var = tf.nn.moments(self.outputs[self.counter - 1], [0, 1, 2, 3])
        self.bn_means += [bn_mean]
        self.bn_vars += [bn_var]
        normalized = tf.nn.batch_normalization(self.outputs[self.counter - 1],
                                               self.bn_means[self.counter], self.bn_vars[self.counter],
                                               offset=None, scale=None, variance_epsilon=1e-6)
        self.weights += [init_weight([filter_size, filter_size, n_input, n_output])]
        self.biases += [init_bias([n_output])]
        convolved = tf.nn.conv2d(normalized, self.weights[self.counter], strides=[1, stride_step, stride_step, 1], padding='SAME')
        self.outputs += lrelu(tf.add(convolved, self.biases[self.counter]))

    def get_layer(self, level):
        return self.outputs[level]

    def get_outputs(self):
        return self.outputs

    def get_weights(self):
        return self.weights


class decoder(object):
    def __init__(self, decoder_inputs, enc):
        self.bn_means = [None] * 6
        self.bn_vars = [None] * 6
        self.weights = [None] * 6
        self.biases = [None] * 6
        self.outputs = [None] * 6
        self.counter = 5
        self.decoder_inputs = decoder_inputs
        self.enc = enc

    def step(self, current_input, stride_step, depth):
        self.counter += -1
        shape = self.enc.get_outputs()[self.counter].get_shape().as_list()
        weight_shape = self.enc.get_weights()[self.counter+1]
        bn_mean, bn_var = tf.nn.moments(current_input, [0, 1, 2, 3])
        self.bn_means[depth] += [bn_mean]
        self.bn_vars[depth] += [bn_var]
        normalized = tf.nn.batch_normalization(current_input,
                                               self.bn_means[depth][self.counter], self.bn_vars[depth][self.counter],
                                               offset=None, scale=None, variance_epsilon=1e-6)
        self.weights[depth] += init_weight(weight_shape)
        bias_shape = self.weights[depth][self.counter].get_shape().as_list()[2]
        self.biases[depth] += init_bias(bias_shape)
        deconvolved = tf.nn.conv2d_transpose(
            normalized, self.weights[depth][self.counter],
            tf.pack([tf.shape(self.x)[0], shape[1], shape[2], shape[3]]),
            strides=[1, stride_step, stride_step, 1], padding='SAME')
        self.outputs[depth] += lrelu(tf.add(deconvolved, self.biases[depth][self.counter]))

    def get_layer(self, depth, level):
        return self.outputs[depth][level]


class fc_layer(object):

    def __init__(self, latent_dim):
        self.latent_dim = latent_dim

    def compress(self, input_matrix):
        self.pre_latent_shape = input_matrix.get_shape().as_list()[1:]
        self.pre_latent_size = np.product(self.pre_latent_shape)
        self.fc_compression = tf.divide(float(self.latent_dim), self.pre_latent_size)
        self.W_fc_in = init_weight([self.pre_latent_size, self.latent_dim])
        self.b_fc_in = init_bias(self.latent_dim)
        input_matrix = tf.reshape(input_matrix, [-1, self.pre_latent_size])
        self.z = lrelu(tf.matmul(input_matrix, self.W_fc_in) + self.b_fc_in)

    def expand(self):
        self.W_fc_out = init_weight([self.latent_dim, self.pre_latent_size])
        self.b_fc_out = init_bias(self.pre_latent_size)
        hidden_step = lrelu(tf.matmul(self.z, self.W_fc_out) + self.b_fc_out)
        self.output = tf.reshape(hidden_step, [-1] + self.pre_latent_shape)

    def expand_arbitrary_vector(self, input_vector):
        self.z = input_vector
        self.expand()

    def execute(self, input_matrix):
        self.compress(input_matrix)
        self.expand()

    def get_output(self):
        return self.output


class autoencoder(object):
    def __init__(self):
        self.input_shape = [None, IMAGE_DIM ** 2]
        self.latent_dim = LATENT_DIM
        self.layers_deep = 6
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

    def build_graph(self):
        if self.corruption:
            self.x_tensor = corrupt(self.x_tensor)  # Optionally apply denoising autoencoder

        enc = encoder(self.x_tensor)
        for layer_i in range(1, 6 * self.duplication):
            n_output = self.n_filters[layer_i]
            enc.step(n_output, self.filter_sizes[layer_i - 1], self.striders[layer_i - 1])

        fc = fc_layer(self.latent_dim)
        fc.execute(enc.get_layer(5))
        decoder_inputs = enc.get_outputs() + fc.get_output()

        dec = decoder(decoder_inputs, enc)
        for layer_i in range(0, (6 * self.duplication) - 1):
            reversed_index = (6 * self.duplication) - 2 - layer_i
            dec.step(self.striders[reversed_index])

        self.y = dec.get_layer(6, 0)
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


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
CONV_DEPTH = 2
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

        self.duplication = 1
        self.sides_dim = [int(IMAGE_DIM * 2 ** -i) for i in range(7)]
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
        init = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(init)

    def init_bias(self, shape):
        return tf.Variable(tf.zeros(shape))


    def fully_connected(self):
        pre_latent_shape = self.enc_outputs[-1].get_shape().as_list()[1:]
        pre_latent_size = np.product(pre_latent_shape)
        self.compression = tf.divide(float(self.latent_dim), pre_latent_size)

        W_fc_in = self.init_weight([pre_latent_size, self.latent_dim])
        b_fc_in = self.init_bias(self.latent_dim)
        W_fc_out = self.init_weight([self.latent_dim, pre_latent_size])
        b_fc_out = self.init_bias(pre_latent_size)
        hidden_1 = tf.reshape(self.enc_outputs[-1], [-1, pre_latent_size])
        self.z = lrelu(tf.matmul(hidden_1, W_fc_in) + b_fc_in)
        hidden_2 = lrelu(tf.matmul(self.z, W_fc_out) + b_fc_out)
        self.fc_out = tf.reshape(hidden_2, [-1] + pre_latent_shape)


    def encoder_step(self, counter, n_input, n_output, filter_size, stride_step):
        self.shapes.append(self.enc_outputs[counter - 1].get_shape().as_list())
        bn_mean, bn_var = tf.nn.moments(self.enc_outputs[counter - 1], [0, 1, 2, 3])
        self.enc_bn_means += [bn_mean]
        self.enc_bn_vars += [bn_var]
        normalized = tf.nn.batch_normalization(self.enc_outputs[counter - 1],
                                               self.enc_bn_means[counter], self.enc_bn_vars[counter],
                                               offset=None, scale=None, variance_epsilon=1e-6)

        self.enc_weights += [self.init_weight([filter_size, filter_size, n_input, n_output])]
        self.enc_biases += [self.init_bias([n_output])]
        convolved = tf.nn.conv2d(normalized, self.enc_weights[counter], strides=[1, 1, 1, 1],
                                 padding='SAME')
        output = lrelu(tf.add(convolved, self.enc_biases[counter]))

        self.enc_weights_2 += [self.init_weight([filter_size, filter_size, n_output, n_output])]
        self.enc_biases_2 += [self.init_bias([n_output])]
        convolved_2 = tf.nn.conv2d(output, self.enc_weights_2[counter], strides=[1, 1, 1, 1],
                                 padding='SAME')
        output_2 = lrelu(tf.add(convolved_2, self.enc_biases_2[counter]))

        # self.shapes_3.append(self.enc_outputs[counter - 1].get_shape().as_list())
        self.enc_weights_3 += [self.init_weight([filter_size, filter_size, n_output, n_output])]
        self.enc_biases_3 += [self.init_bias([n_output])]
        convolved_3 = tf.nn.conv2d(output_2, self.enc_weights_3[counter], strides=[1, 2, 2, 1],
                                 padding='SAME')
        output_3 = lrelu(tf.add(convolved_3, self.enc_biases_3[counter]))

        self.enc_outputs += [output_3]


    def decoder_step(self, depth, counter):
        if counter > depth:
            reversed_index = 5 - counter
            # shape = self.shapes[reversed_index]
            # shape_3 = self.shapes_3[reversed_index]
            # weight_shape_3 = self.enc_weights_3[reversed_index + 1].get_shape()
            # stride_step = self.striders[reversed_index]
            bn_mean, bn_var = tf.nn.moments(self.dec_outputs[depth][counter-1], [0, 1, 2, 3])
            self.dec_bn_means[depth] += [bn_mean]
            self.dec_bn_vars[depth] += [bn_var]
            normalized = tf.nn.batch_normalization(self.dec_outputs[depth][counter-1], # 0, 1, 2, 3, 4
                                                   self.dec_bn_means[depth][counter], self.dec_bn_vars[depth][counter],  # 1, 2, 3, 4, 5
                                                   offset=None, scale=None, variance_epsilon=1e-6)

            # weight_shape = self.enc_weights[reversed_index + 1].get_shape()
            filters_out = self.n_filters[5-counter]
            filters_in = self.n_filters[6-counter]
            kernel_dim = self.filter_sizes[5-counter]
            dim_out = self.sides_dim[5-counter]
            n_batches = tf.shape(self.x)[0]

            self.dec_weights[depth] += [self.init_weight([kernel_dim, kernel_dim, filters_out, filters_in])]
            self.dec_biases[depth] += [self.init_bias(filters_out)]
            deconvolved = tf.nn.conv2d_transpose(
                    normalized, self.dec_weights[depth][counter],
                    tf.pack([n_batches, dim_out, dim_out, filters_out]),
                    strides=[1, 2, 2, 1], padding='SAME')
            output = lrelu(tf.add(deconvolved, self.dec_biases[depth][counter]))

            self.dec_weights_2[depth] += [self.init_weight([kernel_dim, kernel_dim, filters_out, filters_out])]
            self.dec_biases_2[depth] += [self.init_bias(filters_out)]
            deconvolved_2 = tf.nn.conv2d_transpose(
                    output, self.dec_weights_2[depth][counter],
                    tf.pack([n_batches, dim_out, dim_out, filters_out]),
                    strides=[1, 1, 1, 1], padding='SAME')
            output_2 = lrelu(tf.add(deconvolved_2, self.dec_biases_2[depth][counter]))

            self.dec_weights_3[depth] += [self.init_weight([kernel_dim, kernel_dim, filters_out, filters_out])]
            self.dec_biases_3[depth] += [self.init_bias(filters_out)]
            deconvolved_3 = tf.nn.conv2d_transpose(
                    output_2, self.dec_weights_3[depth][counter],
                    tf.pack([n_batches, dim_out, dim_out, filters_out]),
                    strides=[1, 1, 1, 1], padding='SAME')
            output_3 = lrelu(tf.add(deconvolved_3, self.dec_biases_3[depth][counter]))

            self.dec_outputs[depth] += [output_3]
        else:
            self.dec_bn_means[depth] += [None]
            self.dec_bn_vars[depth] += [None]
            self.dec_weights[depth] += [None]
            self.dec_biases[depth] += [None]



    def build_graph(self):
        if self.corruption:
            self.x_tensor = corrupt(self.x_tensor) # Optionally apply denoising autoencoder

        self.enc_weights = [None]
        self.enc_biases = [None]
        self.enc_weights_2 = [None]
        self.enc_biases_2 = [None]
        self.enc_weights_3 = [None]
        self.enc_biases_3 = [None]
        self.enc_bn_means = [None]
        self.enc_bn_vars = [None]
        self.enc_outputs = [self.x_tensor]

        self.shapes = []
        self.shapes_3 = []
        for layer_i in range(1, 6):
            n_output = self.n_filters[layer_i]
            n_input = self.enc_outputs[layer_i - 1].get_shape().as_list()[3]
            self.encoder_step(layer_i, n_input, n_output, self.filter_sizes[layer_i - 1], self.striders[layer_i - 1])

        # self.fully_connected()

        self.dec_weights = [[None]] * 5
        self.dec_biases = [[None]] * 5
        self.dec_weights_2 = [[None]] * 5
        self.dec_biases_2 = [[None]] * 5
        self.dec_weights_3 = [[None]] * 5
        self.dec_biases_3 = [[None]] * 5
        self.dec_bn_means = [[None]] * 5
        self.dec_bn_vars = [[None]] * 5
        self.dec_outputs = [[None]*(i) + [self.enc_outputs[5-i]] for i in range(5)]

        self.ys = []
        self.costs = []
        for depth in range(0, 5):
            for layer_i in range(1, 6):
                self.decoder_step(depth, layer_i)

            self.ys += [self.dec_outputs[depth][-1]]  # Pass the current state from the previous operations
            self.costs += [tf.reduce_sum(tf.square(self.ys[depth] - self.x_tensor))]


#===============================================================================================


ae = autoencoder()
optimizers = [tf.train.AdamOptimizer(LEARNING_RATE).minimize(ae.costs[depth]) for depth in range(5)]

# We create a session to use the graph
tf.Graph().as_default()
sess = tf.Session()
coord = tf.train.Coordinator()
tf.train.start_queue_runners(sess=sess, coord=coord)
sess.run(tf.initialize_all_variables())

# Fit all training data
costs = []
for depth in range(4, -1, -1):
    for epoch_i in range(N_EPOCHS):
        for batch_i in range(1): #TODO
            batch_xs, _ = sess.run([images_batch, labels_batch])
            # batch_xs_f = np.reshape(batch_xs, (BATCH_SIZE, IMAGE_DIM, IMAGE_DIM))
            # batch_xs_f = [np.fft.fftshift(np.fft.fft2(batch_xs_f[i])) for i in range(len(batch_xs))]
            # batch_xs_f = np.reshape(batch_xs, (BATCH_SIZE, IMAGE_DIM * IMAGE_DIM))=
            train = batch_xs
            sess.run(optimizers[depth], feed_dict={ae.x: train})
        if epoch_i % 25 == 0:
            format_epoch = str(epoch_i) #TODO
            cost_reported = sess.run(ae.costs[depth], feed_dict={ae.x: train})
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
recon = sess.run(ae.ys[0], feed_dict={ae.x: test_xs_norm})


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
plt.ylim(10**2, 10**8)
plt.show()


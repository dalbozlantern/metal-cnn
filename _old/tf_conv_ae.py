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
NUM_LAYERS = 6
CONV_DEPTH = 2
BATCH_SIZE = 5
CHANNELS = 1
MIN_AFTER_DEQUEUE = 1000
N_EPOCHS = 100000
LEARNING_RATE = 0.01
LATENT_DIM = 64**2
LAST_DEPTH = 4
print('*'*32 + '\n' + str(int(
    ((IMAGE_DIM/(2**NUM_LAYERS))**2)*LAST_DEPTH
     )) + ' latent size\n' + '*'*32)
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
        self.corruption = True

        self.sides_dim = [int(IMAGE_DIM * 2 ** -i) for i in range(NUM_LAYERS + 1)]
        self.n_filters = [CHANNELS] + [10]*(NUM_LAYERS - 1) + [LAST_DEPTH]
        self.striders = [2] * (NUM_LAYERS + 1)
        self.filter_sizes = [7] + [3]*NUM_LAYERS

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


    def encoder_step(self, counter):

        bn_mean, bn_var = tf.nn.moments(self.enc_outputs[counter - 1], [0, 1, 2, 3])
        self.enc_bn_means += [bn_mean]
        self.enc_bn_vars += [bn_var]
        normalized = tf.nn.batch_normalization(self.enc_outputs[counter - 1],
                                               self.enc_bn_means[counter], self.enc_bn_vars[counter],
                                               offset=None, scale=None, variance_epsilon=1e-6)
        # Initialize repeated convolutions
        outputs = [normalized]
        kernel_dim = self.filter_sizes[counter - 1]
        filters_out = self.n_filters[counter]

        # Iterated convolutions + ReLU; compression on final layer
        for conv in range(CONV_DEPTH):
            if conv == 0:
                filters_in = self.n_filters[counter - 1]
            else:
                filters_in = self.n_filters[counter]
            if conv == CONV_DEPTH - 1:
                stride_len = 2
            else:
                stride_len = 1

            self.enc_weights[conv] += [self.init_weight([kernel_dim, kernel_dim, filters_in, filters_out])]
            self.enc_biases[conv] += [self.init_bias([filters_out])]
            convolved = tf.nn.conv2d(outputs[-1], self.enc_weights[conv][-1], strides=[1, stride_len, stride_len, 1],
                                     padding='SAME')
            outputs += [lrelu(tf.add(convolved, self.enc_biases[conv][-1]))]

        self.enc_outputs += [outputs[-1]]


    def decoder_step(self, depth, counter):
        if counter > depth:

            bn_mean, bn_var = tf.nn.moments(self.dec_outputs[depth][counter-1], [0, 1, 2, 3])
            self.dec_bn_means[depth] += [bn_mean]
            self.dec_bn_vars[depth] += [bn_var]
            normalized = tf.nn.batch_normalization(self.dec_outputs[depth][counter-1],
                                                   self.dec_bn_means[depth][counter], self.dec_bn_vars[depth][counter],
                                                   offset=None, scale=None, variance_epsilon=1e-6)

            # Initialize repeated deconvolutions
            outputs = [normalized]
            kernel_dim = self.filter_sizes[NUM_LAYERS - counter]
            dim_out = self.sides_dim[NUM_LAYERS - counter]
            n_batches = tf.shape(self.x)[0]
            filters_out = self.n_filters[NUM_LAYERS - counter]

            # Iterated convolutions + ReLU; decompression on first layer
            for conv in range(CONV_DEPTH):
                if conv == 0:
                    filters_in = self.n_filters[NUM_LAYERS + 1 - counter]
                    stride_len = 2
                else:
                    filters_in = self.n_filters[NUM_LAYERS - counter]
                    stride_len = 1
                self.dec_weights[conv][depth] += [self.init_weight([kernel_dim, kernel_dim, filters_out, filters_in])]
                self.dec_biases[conv][depth] += [self.init_bias(filters_out)]
                deconvolved = tf.nn.conv2d_transpose(
                        outputs[-1], self.dec_weights[conv][depth][-1],
                        tf.pack([n_batches, dim_out, dim_out, filters_out]),
                        strides=[1, stride_len, stride_len, 1], padding='SAME')
                outputs += [lrelu(tf.add(deconvolved, self.dec_biases[conv][depth][-1]))]

            self.dec_outputs[depth] += [outputs[-1]]
        else:
            self.dec_bn_means[depth] += [None]
            self.dec_bn_vars[depth] += [None]
            for conv in range(CONV_DEPTH):
                self.dec_weights[conv][depth] += [None]
                self.dec_biases[conv][depth] += [None]



    def build_graph(self):
        if self.corruption:
            self.x_tensor = corrupt(self.x_tensor) # Optionally apply denoising autoencoder

        self.enc_bn_means = [None]
        self.enc_bn_vars = [None]
        self.enc_weights = [[None]] * CONV_DEPTH
        self.enc_biases = [[None]] * CONV_DEPTH
        self.enc_outputs = [self.x_tensor]

        for layer_i in range(1, NUM_LAYERS + 1):
            self.encoder_step(layer_i)

        # self.fully_connected()

        self.dec_bn_means = [[None]] * NUM_LAYERS
        self.dec_bn_vars = [[None]] * NUM_LAYERS
        self.dec_weights = [[[None]] * NUM_LAYERS] * CONV_DEPTH
        self.dec_biases = [[[None]] * NUM_LAYERS] * CONV_DEPTH
        self.dec_outputs = [[None]*(i) + [self.enc_outputs[NUM_LAYERS-i]] for i in range(NUM_LAYERS)]

        self.ys = []
        self.costs = []
        for depth in range(0, NUM_LAYERS):
            for layer_i in range(1, NUM_LAYERS + 1):
                self.decoder_step(depth, layer_i)

            self.ys += [self.dec_outputs[depth][-1]]  # Pass the current state from the previous operations
            self.costs += [tf.reduce_sum(tf.square(self.ys[depth] - self.x_tensor))]


#===============================================================================================


ae = autoencoder()
optimizers = [tf.train.AdamOptimizer(LEARNING_RATE).minimize(ae.costs[depth]) for depth in range(NUM_LAYERS)]

# We create a session to use the graph
tf.Graph().as_default()
sess = tf.Session()
coord = tf.train.Coordinator()
tf.train.start_queue_runners(sess=sess, coord=coord)
sess.run(tf.initialize_all_variables())

# Fit all training data
costs = []
for depth in range(NUM_LAYERS - 1, -1, -1):
    for epoch_i in range(N_EPOCHS):
        for batch_i in range(1): #TODO
            batch_xs, _ = sess.run([images_batch, labels_batch])
            train = batch_xs
            sess.run(optimizers[depth], feed_dict={ae.x: train})
        if epoch_i % 25 == 0:
            format_epoch = str(epoch_i) #TODO
            cost_reported = sess.run(ae.costs[depth], feed_dict={ae.x: train})
            format_cost = str(cost_reported) #TODO
            info_string = 'Depth: ' + str(NUM_LAYERS - depth) + '/' + str(NUM_LAYERS) + ' | Epoch: ' + format_epoch + ' | Cost: ' + format_cost
            progress_bar(info_string, epoch_i, N_EPOCHS)  #TODO
            costs += [cost_reported]

    # Plot example reconstructions
    n_examples = 5
    test_xs, _ = sess.run([images_batch, labels_batch])

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


plt.plot(costs)
plt.xscale('log')
plt.yscale('log')
plt.ylim(10**2, 10**8)
plt.show()


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
MIN_AFTER_DEQUEUE = 3000
N_EPOCHS = 1000
LEARNING_RATE = 0.01
LATENT_DIM = 32
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
# class autoencoder(object):
def autoencoder(input_shape=[None, IMAGE_DIM ** 2],
                latent_dim=LATENT_DIM,
                # n_filters=[1, 10, 10] + [10, 10, 10]*4 + [3, 3, 3],
                # striders=[2, 1, 1] * 6,
                # filter_sizes=[7, 3, 3] + [3, 3, 3]*5,
                n_filters=[1, 10] + [10, 10]*4 + [3, 3],
                striders=[2, 1] * 6,
                filter_sizes=[7, 3] + [3, 3]*5,
                # n_filters=[1] + [10]*4 + [3],
                # striders=[2] * 6,
                # filter_sizes=[7] + [3]*5,
                corruption=False):

    # Input to the network
    x = tf.placeholder(
        tf.float32, input_shape, name='x')

    # Ensure 2-d is converted to square tensor.
    if len(x.get_shape()) == 2:
        x_dim = np.sqrt(x.get_shape().as_list()[1])
        if x_dim != int(x_dim):
            raise ValueError('Unsupported input dimensions')
        x_dim = int(x_dim)
        x_tensor = tf.reshape(
            x, [-1, x_dim, x_dim, n_filters[0]])
    elif len(x.get_shape()) == 4:
        x_tensor = x
    else:
        raise ValueError('Unsupported input dimensions')
    current_input = x_tensor

    # Optionally apply denoising autoencoder
    if corruption:
        current_input = corrupt(current_input)

    # Build the encoder
    encoder = []
    shapes = []
    counter = 0
    for layer_i, n_output in enumerate(n_filters[1:]):
        n_input = current_input.get_shape().as_list()[3]
        shapes.append(current_input.get_shape().as_list())
        W = tf.Variable(
            tf.random_uniform([
                filter_sizes[layer_i],
                filter_sizes[layer_i],
                n_input, n_output],
                -1.0 / math.sqrt(n_input),
                1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        bn_mean, bn_var = tf.nn.moments(current_input, [0, 1, 2, 3])
        hidden_step = tf.nn.batch_normalization(current_input, bn_mean, bn_var, offset=None, scale=None, variance_epsilon=1e-6)
        output = lrelu(
            tf.add(tf.nn.conv2d(
                hidden_step, W, strides=[1, striders[layer_i], striders[layer_i], 1], padding='SAME'), b))
        current_input = output

    # Fully connected
    pre_latent_shape = current_input.get_shape().as_list()[1:]
    pre_latent_size = np.product(pre_latent_shape)
    compression = tf.divide(float(latent_dim), pre_latent_size)

    W_fc_in = tf.Variable(tf.random_uniform([pre_latent_size, latent_dim], -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input)))
    b_fc_in = tf.Variable(tf.zeros(latent_dim))
    W_fc_out = tf.Variable(tf.random_uniform([latent_dim, pre_latent_size], -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input)))
    b_fc_out = tf.Variable(tf.zeros(pre_latent_size))
    current_input = tf.reshape(current_input, [-1, pre_latent_size])
    z = lrelu(tf.matmul(current_input, W_fc_in) + b_fc_in)
    current_input = lrelu(tf.matmul(z, W_fc_out) + b_fc_out)
    current_input = tf.reshape(current_input, [-1] + pre_latent_shape)

    # Store the latent representation
    encoder.reverse()
    shapes.reverse()

    # Build the decoder using the same weights
    for layer_i, shape in enumerate(shapes):
        # W = encoder[layer_i]
        W = tf.Variable(
            tf.random_uniform(encoder[layer_i].get_shape(),
                -1.0 / math.sqrt(n_input),
                1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
        bn_mean, bn_var = tf.nn.moments(current_input, [0, 1, 2, 3])
        hidden_step = tf.nn.batch_normalization(current_input, bn_mean, bn_var, offset=None, scale=None, variance_epsilon=1e-6)
        output = lrelu(tf.add(
            tf.nn.conv2d_transpose(
                hidden_step, W,
                tf.pack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                strides=[1, striders[layer_i], striders[layer_i], 1], padding='SAME'), b))
        current_input = output

    # Now have the reconstruction through the network
    y = current_input
    # Cost function measures pixel-wise difference
    cost = tf.reduce_sum(tf.square(y - x_tensor))

    return {'x': x, 'y': y, 'cost': cost}


ae = autoencoder()
optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(ae['cost'])

# We create a session to use the graph
tf.Graph().as_default()
sess = tf.Session()
coord = tf.train.Coordinator()
tf.train.start_queue_runners(sess=sess, coord=coord)
sess.run(tf.initialize_all_variables())

# Fit all training data
costs = []
for epoch_i in range(N_EPOCHS):
    for batch_i in range(1): #TODO
        batch_xs, _ = sess.run([images_batch, labels_batch])
        # batch_xs_f = np.reshape(batch_xs, (BATCH_SIZE, IMAGE_DIM, IMAGE_DIM))
        # batch_xs_f = [np.fft.fftshift(np.fft.fft2(batch_xs_f[i])) for i in range(len(batch_xs))]
        # batch_xs_f = np.reshape(batch_xs, (BATCH_SIZE, IMAGE_DIM * IMAGE_DIM))=
        train = batch_xs
        sess.run(optimizer, feed_dict={ae['x']: train})
    if epoch_i % 25 == 0:
        format_epoch = str(epoch_i) #TODO
        cost_reported = sess.run(ae['cost'], feed_dict={ae['x']: train})
        format_cost = str(cost_reported) #TODO
        info_string = 'Epoch: ' + format_epoch + ' | Cost: ' + format_cost
        progress_bar(info_string, epoch_i, N_EPOCHS)  #TODO
        costs += [cost_reported]

# Plot example reconstructions
n_examples = 5
test_xs, _ = sess.run([images_batch, labels_batch])
# test_xs_f = np.reshape(test_xs, (BATCH_SIZE, IMAGE_DIM, IMAGE_DIM))
# test_xs_f = [np.fft.fftshift(np.fft.fft2(test_xs_f[i])) for i in range(len(test_xs_f))]
# test_xs_f = np.reshape(test_xs_f, (BATCH_SIZE, IMAGE_DIM * IMAGE_DIM))

test_xs_norm = test_xs #new
recon = sess.run(ae['y'], feed_dict={ae['x']: test_xs_norm})


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


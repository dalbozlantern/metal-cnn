import tensorflow as tf
import numpy as np

from configparser import ConfigParser
config = ConfigParser()
config.read('config.ini')
resized_root = config.get('main', 'resized_root')

with open(resized_root + '/' + 'file_name_queue.txt', 'r') as file:
    file_name_queue = file.read().splitlines()

band_name_matrices = np.load(resized_root + '/' + 'name_matrices.npy')


# INITIALIZE PLACEHOLDERS
x = tf.placeholder(tf.float32, shape=[None, 256*256])
y_actual = tf.placeholder(tf.float32, shape=[None, 20*39])

x_image = tf.reshape(x, [-1, 256, 256, 1])
y_matrix = tf.reshape(y_actual, [-1, 20, 29, 1])

dropout_keep_probability = tf.placeholder(tf.float32)


def build_graph(x, dropout_keep_probability):

    def initialize_weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def initialize_bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def build_layer_1(x, W1, b1, W2, b2):
        hs1 = tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='SAME')
        hs1r = tf.nn.relu(hs1 + b1)
        hs2 = tf.nn.conv2d(hs1r, W2, strides=[1, 1, 1, 1], padding='SAME')
        hs2r = tf.nn.relu(hs2 + b2)
        hs3 = hs2r  # Placeholder for batch norm
        return hs3

    def build_layers_2_plus(x, W1, b1, W2, b2, W3, b3):
        hs0 = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        hs1 = tf.nn.conv2d(hs0, W1, strides=[1, 1, 1, 1], padding='SAME')
        hs1r = tf.nn.relu(hs1 + b1)
        hs2 = tf.nn.conv2d(hs1r, W2, strides=[1, 1, 1, 1], padding='SAME')
        hs2r = tf.nn.relu(hs2 + b2)
        hs3 = tf.nn.conv2d(hs2r, W3, strides=[1, 1, 1, 1], padding='SAME')
        hs3r = tf.nn.relu(hs3 + b3)
        hs4 = hs3r  # Placeholder for batch norm
        return hs4

    # Conv: Input ([256 x 256 x 1]) --> [256 x 256 x 64]
    W1_l1 = initialize_weight_variable([7, 7, 1, 64])
    b1_l1 = initialize_bias_variable([64])
    W2_l1 = initialize_weight_variable([3, 3, 64, 64])
    b2_l1 = initialize_bias_variable([64])
    layer_1_out = build_layer_1(x, W1_l1, b1_l1, W2_l1, b2_l1)

    # Conv: [256 x 256 x 64] --> [128 x 128 x 128]
    W1_l2 = initialize_weight_variable([3, 3, 64, 128])
    b1_l2 = initialize_bias_variable([128])
    W2_l2 = initialize_weight_variable([3, 3, 128, 128])
    b2_l2 = initialize_bias_variable([128])
    W3_l2 = initialize_weight_variable([3, 3, 128, 128])
    b3_l2 = initialize_bias_variable([128])
    layer_2_out = build_layers_2_plus(layer_1_out, W1_l2, b1_l2, W2_l2, b2_l2, W3_l2, b3_l2)

    # Conv: [128 x 128 x 128] --> [64 x 64 x 256]
    W1_l3 = initialize_weight_variable([3, 3, 128, 256])
    b1_l3 = initialize_bias_variable([256])
    W2_l3 = initialize_weight_variable([3, 3, 256, 256])
    b2_l3 = initialize_bias_variable([256])
    W3_l3 = initialize_weight_variable([3, 3, 256, 256])
    b3_l3 = initialize_bias_variable([256])
    layer_3_out = build_layers_2_plus(layer_2_out, W1_l3, b1_l3, W2_l3, b2_l3, W3_l3, b3_l3)

    # Conv: [64 x 64 x 256] --> [32 x 32 x 512]
    W1_l4 = initialize_weight_variable([3, 3, 64, 256])
    b1_l4 = initialize_bias_variable([512])
    W2_l4 = initialize_weight_variable([3, 3, 512, 512])
    b2_l4 = initialize_bias_variable([512])
    W3_l4 = initialize_weight_variable([3, 3, 512, 512])
    b3_l4 = initialize_bias_variable([512])
    layer_4_out = build_layers_2_plus(layer_3_out, W1_l4, b1_l4, W2_l4, b2_l4, W3_l4, b3_l4)

    # Conv: [32 x 32 x 512] --> [16 x 16 x 512]
    W1_l5 = initialize_weight_variable([3, 3, 512, 512])
    b1_l5 = initialize_bias_variable([512])
    W2_l5 = initialize_weight_variable([3, 3, 512, 512])
    b2_l5 = initialize_bias_variable([1024])
    W3_l5 = initialize_weight_variable([3, 3, 512, 512])
    b3_l5 = initialize_bias_variable([1024])
    layer_5_out = build_layers_2_plus(layer_4_out, W1_l5, b1_l5, W2_l5, b2_l5, W3_l5, b3_l5)

    # Conv: [16 x 16 x 512] --> [8 x 8 x 512]
    W1_l6 = initialize_weight_variable([3, 3, 512, 512])
    b1_l6 = initialize_bias_variable([512])
    W2_l6 = initialize_weight_variable([3, 3, 512, 512])
    b2_l6 = initialize_bias_variable([1024])
    W3_l6 = initialize_weight_variable([3, 3, 512, 512])
    b3_l6 = initialize_bias_variable([1024])
    layer_6_out = build_layers_2_plus(layer_5_out, W1_l6, b1_l6, W2_l6, b2_l6, W3_l6, b3_l6)

    # FC: [8 x 8 x 512] ~~ [8*8*512] --> [1024]
    W1_l7 = initialize_weight_variable([8*8*512, 1024])
    b1_l7 = initialize_bias_variable([1024])
    layer_7_in = tf.reshape(layer_6_out, [-1, 8*8*512])
    layer_7_hid = tf.nn.relu(tf.matmul(layer_7_in, W1_l7) + b1_l7)
    layer_7_out = layer_7_hid  # placeholder for batch norm
    layer_7_dropout = tf.nn.dropout(layer_7_out, dropout_keep_probability)

    # FC: [1024] --> [20]  (about the same % contraction as layer 7)
    W1_l8 = initialize_weight_variable([1024, 20])
    b1_l8 = initialize_bias_variable([20])
    layer_8_hid = tf.nn.relu(tf.matmul(layer_7_dropout, W1_l8) + b1_l8)
    layer_8_out = layer_8_hid  # placeholder for batch norm

    # 1x1 conv: [20, 1, 1] --> [20, 39, 1], i.e. (char_position, char_identity, 1)
    W1_l9 = initialize_weight_variable([1, 1, 1, 39])
    b1_l9 = initialize_bias_variable([39])
    layer_9_in = tf.reshape(layer_8_out, [-1, 20, 1, 1])
    layer_9_hid = tf.nn.conv2d(layer_9_in, W1_l9, strides=[1, 1, 1, 1], padding='SAME')
    layer_9_out = tf.nn.relu(layer_9_hid + b1_l9)

    return layer_9_out


def train_network(x, hyperparams):

    # Model incorporation
    predictions = build_graph(x)
        # Input is a [batches=TBD, height=256, width=256, channel=1] tensor
        # Output is a [batches=TBD, char_position=20, char_identity=39, N/A=1] tensor
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_matrix, predictions, 2))
    train_step = tf.train.AdamOptimizer().minimize(cost)
    correct_prediction = tf.equal(tf.argmax(y_matrix, 2), tf.argmax(predictions, 2))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Running
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        print('Training...')
        for epoch_num in range(hyperparams['hm_epochs']):

            for batch_num in range(hyperparams['num_batches']):
                batch = #TODO
                valid_batch =  #TODO

                if batch_num % hyperparams['print_every'] * 15 == 0:
                    display_output_header()

                if batch_num % hyperparams['print_every'] == 0:
                    train_cost = cost.eval(feed_dict={x: batch[0], y_actual: batch[1],
                                                      dropout_keep_probability: hyperparams['train_dropout_keep']
                                                      })
                    train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_actual: batch[1],
                                                              dropout_keep_probability: hyperparams['train_dropout_keep']
                                                              })
                    valid_cost = cost.eval(feed_dict={x: valid_batch[0], valid_batch: batch[1],
                                                      dropout_keep_probability: 1.0
                                                      })
                    valid_accuracy = accuracy.eval(feed_dict={x: valid_batch[0], valid_batch: batch[1],
                                                              dropout_keep_probability: 1.0
                                                              })
                    display_output_line(epoch_num, batch_num, train_cost, train_accuracy, valid_cost, valid_accuracy)

    # TODO: saving params
    # TODO: keeping running tabs / graphing




def display_output_header():
    print('|=========================================================|\n' +
          '|       |       |      Training      |     Validation     |\n' +
          '| Epoch | Batch |  Cost   | Accuracy |  Cost   | Accuracy |\n' +
          '|---------------------------------------------------------|')


def display_output_line(epoch_num, batch_num, train_cost, train_accuracy, valid_cost, valid_accuracy):

    def space_format_int(str_in):
        str_in = str(str_in)
        num_spaces = 5 - len(str_in)
        return ' ' * num_spaces + str_in

    def space_format_percent(float_in):
        percent = '{0:.1f}%'.format(float_in * 100)
        return space_format_int(percent)

    def space_format_scientific(float_in):
        return '{:.1e}'.format(float_in)

    print('| ' + space_format_int(epoch_num) + \
          ' | ' + space_format_int(batch_num) + \
          ' | ' + space_format_scientific(train_cost) + \
          ' |  ' + space_format_percent(train_accuracy) + \
          '   | ' + space_format_scientific(valid_cost) + \
          ' |  ' + space_format_percent(valid_accuracy) + '   |)')





hyperparams = {}
hyperparams['hm_epochs'] = 10
hyperparams['num_batches'] =  # TODO
hyperparams['train_dropout_keep'] = .7
hyperparams['print_every'] = 100

train_network(x, hyperparams)


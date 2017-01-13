import tensorflow as tf
import numpy as np

from configparser import ConfigParser
config = ConfigParser()
config.read('config.ini')
resized_root = config.get('main', 'resized_root')



with tf.Graph().as_default():

    hyperparams = {'hm_epochs': int(73000/50),
                   'train_dropout_keep': .7,
                   'print_every': 1,
                   'batch_size': 50,
                   'min_after_dequeue': 1000,
                   'img_size': 256,
                   }
    hyperparams['num_batches'] = 1  # TODO




    # INITIALIZE PLACEHOLDERS
    x = tf.placeholder(tf.float32, shape=[None, hyperparams['img_size'] ** 2])
    y_actual = tf.placeholder(tf.float32, shape=[None, 20*39])

    x_image = tf.reshape(x, [-1, 256, 256, 1])
    y_matrix = tf.reshape(y_actual, [-1, 20, 1, 39])

    dropout_keep_probability = tf.placeholder(tf.float32)


    def build_graph(x, dropout_keep_probability):

        def initialize_weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def initialize_bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        def conv_with_relu(x, in_depth, out_depth):
            weights = initialize_weight_variable([3, 3, in_depth, out_depth])
            bias = initialize_bias_variable([out_depth])
            hidden_step = tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME')
            return tf.nn.relu(hidden_step + bias)


        def encoder_layer(x, in_depth, out_depth, pool=True, conv=1):
            if pool:
                hidden_step_0 = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            else:
                hidden_step_0 = x
            hidden_step_1 = conv_with_relu(hidden_step_0, in_depth, out_depth)
            if conv == 2:
                hidden_step_2 = conv_with_relu(hidden_step_1, out_depth, out_depth)
            else:
                hidden_step_2 = hidden_step_1
            bn_mean, bn_var = tf.nn.moments(hidden_step_2, [0, 1, 2, 3])
            return tf.nn.batch_normalization(hidden_step_2, bn_mean, bn_var, offset=None, scale=None, variance_epsilon=1e-6)


        layer_1_out = encoder_layer(x, 1, 64, pool=False)  # Conv: Input ([256 x 256 x 1]) --> [256 x 256 x 64]
        layer_2_out = encoder_layer(layer_1_out, 64, 128)  # Conv: [256 x 256 x 64] --> [128 x 128 x 128]
        layer_3_out = encoder_layer(layer_2_out, 128, 256)  # Conv: [128 x 128 x 128] --> [64 x 64 x 256]
        layer_4_out = encoder_layer(layer_3_out, 256, 512)  # Conv: [64 x 64 x 256] --> [32 x 32 x 512]
        layer_5_out = encoder_layer(layer_4_out, 512, 512)  # Conv: [32 x 32 x 512] --> [16 x 16 x 512]
        layer_6_out = encoder_layer(layer_5_out, 512, 512)  # Conv: [16 x 16 x 512] --> [8 x 8 x 512]


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
              ' |  ' + space_format_percent(valid_accuracy) + '   |')


    def read_and_decode_input_pair(file_name, img_size):
        file_name_queue = tf.train.string_input_producer([file_name], num_epochs=None)
        reader = tf.TFRecordReader()
        _, serialized_pair = reader.read(file_name_queue)
        components = tf.parse_single_example(serialized_pair,
                                             features={'label_matrix': tf.FixedLenFeature([20*39], tf.float32),
                                                       'image': tf.FixedLenFeature([img_size**2], tf.float32)})
        image = components['image']
        image = tf.div(image, 255)
        label_matrix = components['label_matrix']

        return image, label_matrix


    file_name = resized_root + '/' + str(hyperparams['img_size']) + '_images_and_names.tfrecords'
    image, label_matrix = read_and_decode_input_pair(file_name, hyperparams['img_size'])

    images_batch, labels_batch = tf.train.shuffle_batch(
        [image, label_matrix],
        batch_size=hyperparams['batch_size'],
        capacity=(3 * hyperparams['min_after_dequeue'] + hyperparams['batch_size']),
        min_after_dequeue=hyperparams['min_after_dequeue']
    )

    # Model incorporation
    predictions = build_graph(x_image, hyperparams['train_dropout_keep'])
        # Input is a [batches=TBD, height=256, width=256, channel=1] tensor
        # Output is a [batches=TBD, char_position=20, char_identity=39, N/A=1] tensor
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predictions, y_matrix, 3))
    softmaxes = tf.nn.softmax(predictions)
    cost = tf.reduce_mean(tf.nn.l2_loss(tf.subtract(y_matrix, softmaxes)))
    train_step = tf.train.AdamOptimizer().minimize(cost)
    correct_prediction = tf.equal(tf.argmax(y_matrix, 3), tf.argmax(predictions, 3))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #TODO

    # Running
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        coord = tf.train.Coordinator()

        tf.train.start_queue_runners(sess=sess, coord=coord)

        print('\n\n\nTraining...')
        for epoch_num in range(hyperparams['hm_epochs']):

            for batch_num in range(hyperparams['num_batches']):
                x_qq, y_actual_qq = sess.run([images_batch, labels_batch])
                # valid_batch =  #TODO

                # if batch_num == 0:
                #     display_output_header()

                if batch_num % hyperparams['print_every'] == 0:
                    train_cost = cost.eval(feed_dict={x: x_qq, y_actual: y_actual_qq,
                                                      dropout_keep_probability: hyperparams['train_dropout_keep']
                                                      })
                    train_accuracy = accuracy.eval(feed_dict={x: x_qq, y_actual: y_actual_qq,
                                                              dropout_keep_probability: hyperparams['train_dropout_keep']
                                                              })
                    # TODO
                    # valid_cost = cost.eval(feed_dict={x: valid_batch[0], valid_batch: batch[1],
                    #                                   dropout_keep_probability: 1.0
                    #                                   })
                    # valid_accuracy = accuracy.eval(feed_dict={x: valid_batch[0], valid_batch: batch[1],
                    #                                           dropout_keep_probability: 1.0
                    #                                           })
                    valid_cost = 0
                    valid_accuracy = 0
                    display_output_line(epoch_num, batch_num, train_cost, train_accuracy, valid_cost, valid_accuracy)

                train_step.run(feed_dict={x: x_qq, y_actual: y_actual_qq,
                                          dropout_keep_probability: hyperparams['train_dropout_keep']
                                          })

    # TODO: saving params
    # TODO: keeping running tabs / graphing

    coord.request_stop()
    # coord.join([t])  TODO


    # # SCRATCHWORK
    # fd_in = {x: x_qq, y_actual: y_actual_qq, dropout_keep_probability: hyperparams['train_dropout_keep']}
    # def qth(arg):
    #     return arg.eval(feed_dict=fd_in)
    # pred = qth(predictions)
    # y_m = qth(y_matrix)
    # qth(tf.argmax(y_matrix, 3)).shape
import tensorflow as tf




x = tf.placeholder(tf.float32, shape=[None, 256*256])
y_actual = tf.placeholder(tf.float32, shape=[None, 20*39])

x_image = tf.reshape(x, [-1, 256, 256, 1])
y_matrix = tf.reshape(y_actual, [-1, 20, 29, 1])


def build_graph(x):

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

    # FC: [1024] --> [20]  (about the same % contraction as layer 7)
    W1_l8 = initialize_weight_variable([1024, 20])
    b1_l8 = initialize_bias_variable([20])
    layer_8_hid = tf.nn.relu(tf.matmul(layer_7_out, W1_l8) + b1_l8)
    layer_8_out = layer_8_hid  # placeholder for batch norm

    # 1x1 conv: [20, 1, 1] --> [20, 39, 1], i.e. (char_position, char_identity, 1)
    W1_l9 = initialize_weight_variable([1, 1, 1, 39])
    b1_l9 = initialize_bias_variable([39])
    layer_9_in = tf.reshape(layer_8_out, [-1, 20, 1, 1])
    layer_9_hid = tf.nn.conv2d(layer_9_in, W1_l9, strides=[1, 1, 1, 1], padding='SAME')
    layer_9_out = tf.nn.relu(layer_9_hid + b1_l9)

    return layer_9_out


def train_network(x):

    # Model incorporation
    predictions = build_graph(x)
        # Input is a [batches=TBD, height=256, width=256, channel=1] tensor
        # Output is a [batches=TBD, char_position=20, char_identity=39, N/A=1] tensor
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_matrix, predictions, 2))
    train_step = tf.train.AdamOptimizer().minimize(cost)
    correct_prediction = tf.equal(tf.argmax(y_matrix, 2), tf.argmax(predictions, 2))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Additional hyperparameters
    hm_epochs = 10
    num_batches = #TODO

    # Running
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        print('Training...')
        for epoch in range(hm_epochs):
            epoch_loss = 0

            for batch_num in range(num_batches):
                batch = #TODO mnist.train.next_batch(batch_size)

                if batch_num % 100 == 0:
                    train_accuracy = accuracy.eval(feed_dict={
                        x: batch[0], y_actual: batch[1], dropout_keep_probabilities: 1.0
                    })
                    print('>>>> Epoch %d, step %d: training accuracy %g' % (epoch, batch_num, train_accuracy))

    print('Evaluating accuracy on the test set...')
    test_batch = #TODO
    test_accuracy = accuracy.eval(feed_dict={
        x: test_batch[0], y_actual: test_batch[1], dropout_keep_probabilities: 1.0
    })
    print('>>>> Test accuracy: %g' % test_accuracy)
import tensorflow as tf
import numpy as np

from configparser import ConfigParser
config = ConfigParser()
config.read('config.ini')
resized_root = config.get('main', 'resized_root')


batch_size = 5
min_after_dequeue = 1000
image_dim = 256
file_name = resized_root + '/' + str(image_dim) + '_images_and_names.tfrecords'


# def initialize_batches(file_name, image_dim, min_after_dequeue, batch_size):

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

image, label_matrix = read_and_decode_input_pair(file_name, image_dim)
images_batch, labels_batch = tf.train.shuffle_batch(
    [image, label_matrix],
    batch_size=batch_size,
    capacity=(3 * min_after_dequeue + batch_size),
    min_after_dequeue=min_after_dequeue
)


tf.Graph().as_default()
sess = tf.Session()
coord = tf.train.Coordinator()
tf.train.start_queue_runners(sess=sess, coord=coord)


x, y = sess.run([images_batch, labels_batch])


from utils.greyscaling import show_image
show_image(np.reshape(x[0], (256, 256))*255)
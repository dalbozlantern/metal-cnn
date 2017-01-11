import tensorflow as tf
resized_root = 'a'

def read_images_from_disk(input_queue):
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_png(file_contents, channels=3)
    return example, label

with open(resized_root + '/' + 'file_name_queue.txt', 'r') as file:
    file_name_queue = file.read().splitlines()
label_list =  #TODO

images = tf.python.ops.convert_to_tensor(file_name_queue, dtype=dtypes.string)
labels = tf.python.ops.convert_to_tensor(label_list, dtype=dtypes.int32)

# Makes an input queue
input_queue = tf.train.slice_input_producer([images, labels],
                                            num_epochs=num_epochs,
                                            shuffle=True)

image, label = read_images_from_disk(input_queue)

# Optional Image and Label Batching
image_batch, label_batch = tf.train.batch([image, label],
                                          batch_size=batch_size)








import tensorflow as tf
from configparser import ConfigParser
config = ConfigParser()
config.read('config.ini')
resized_root = config.get('main', 'resized_root')


img_size = 256
file_name = resized_root + '/' + str(img_size) + '_images_and_names.tfrecords'
for serialized_example in tf.python_io.tf_record_iterator(file_name):
    example = tf.train.Example()
    example.ParseFromString(serialized_example)

    image = example.features.feature['image'].float_list.value
    image = tf.cast(image, tf.float32) / 255
    image
    label_matrix = example.features.feature['label_matrix'].float_list.value
    break




#
hyperparams = {'batch_size': 128,
               'min_after_dequeue': 1000}
#

import tensorflow as tf
import numpy as np

from configparser import ConfigParser
config = ConfigParser()
config.read('config.ini')
resized_root = config.get('main', 'resized_root')


def read_and_decode_input_pair(file_name, img_size):
    file_name_queue = tf.train.string_input_producer([file_name], num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_pair = reader.read(file_name_queue)
    components = tf.parse_single_example(serialized_pair,
                                         features={'label_matrix': tf.FixedLenFeature([20*39]),
                                                   'image': tf.FixedLenFeature[img_size**2]})
    image = np.resize(components['image'], (img_size, img_size))
    label_matrix = np.resize(components['label_matrix'], (20, 39))

    return image, label_matrix

img_size = 256
file_name = resized_root + '/' + str(img_size) + '_images_and_names.tfrecords'
image, label_matrix = read_and_decode_input_pair(file_name, img_size)

images_batch, labels_batch = tf.train.shuffle_batch(
    [image, label_matrix],
    batch_size=hyperparams['batch_size'],
    capacity=(3*hyperparams['min_after_dequeue'] + hyperparams['batch_size']),
    min_after_dequeue=hyperparams['min_after_dequeue']
)

sess=tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
tf.train.start_queue_runners(sess=sess)
x, y_matrix = sess.run([images_batch, labels_batch])
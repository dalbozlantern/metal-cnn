import pandas as pd
import csv
import numpy as np
from utils.greyscaling import plot_arbitrary_array
from utils.utils import progress_bar


import tensorflow as tf
from tqdm import tqdm
from scipy import misc

from configparser import ConfigParser
config = ConfigParser()
config.read('config.ini')
resized_root = config.get('main', 'resized_root')


def analyze_band_names(resized_df):
    band_names = list(resized_df['Band'])
    band_name_lengths = [len(name) for name in band_names]
    band_name_lengths.sort()
    plot_arbitrary_array(band_name_lengths)


def clean_df_names(resized_df):
    band_names = list(resized_df['Band'])

    # Import the list of find-and-replace accent marks
    accent_dict = {rows[0]: rows[1] for rows in
                   csv.reader(open('utils/latin_accents.csv', mode='r'))}

    redundant_punctuation = ['\\',
                             '/',
                             "'",
                             '"',
                             '!',
                             '#',
                             '*',
                             '%',
                             '(',
                             ')',
                             ',',
                             '-',
                             '?',
                             ':',
                             ';',
                             '.',
                             '…',
                             '+',
                             '>',
                             '<',
                             '$',
                             ]  # omits & as this might be structurally important

    def sanitize_name(name, trim_len=False):
        for accented_letter, accented_letter_replacement in accent_dict.items():
            name = name.replace(accented_letter, accented_letter_replacement)
        name = name.upper()
        for mark in redundant_punctuation:
            name = name.replace(mark, '')
        name = name.replace('\n', ' ')
        name = name.replace('  ', ' ')
        if trim_len:
            name = name[:trim_len]
        return name

    cleaned_names = [sanitize_name(name, 20) for name in band_names]
    resized_df['Cleaned name'] = cleaned_names
    return resized_df


def analyze_cleaned_names(resized_df):
    cleaned_band_names = list(resized_df['Cleaned name'])
    cleaned_band_name_lengths = [len(name) for name in cleaned_band_names]
    cleaned_band_name_lengths.sort()
    plot_arbitrary_array(cleaned_band_name_lengths)


def build_name_matrices(resized_df):
    #Buld list of valid characters
    char_array1 = [chr(i) for i in range(65, 91)]  # A-Z
    char_array2 = [str(i) for i in range(10)]
    char_array = char_array1 + char_array2 + ['&', ' ']

    num_entries = resized_df.shape[0]
    names_bat = np.empty((num_entries, 20, 39))

    for i, row in resized_df.iterrows():

        if i % 100 == 0:
            progress_bar('Processing image #', i, num_entries)

        cleaned_name = resized_df['Cleaned name'][i]
        num_chars = len(cleaned_name)
        char_matrix = np.empty([20, 39], 'float32')

        for char_index in range(num_chars):
            char = cleaned_name[char_index]
            one_hot = [float(char == char_ID) for char_ID in char_array]
            one_hot += [0.0]  # For the "no char" index
            char_matrix[char_index] = one_hot

        for char_index in range(num_chars, 20):
            char_matrix[char_index] = [0.0] * 38 + [1.0]

        names_bat[i] = char_matrix
    print('\n')
    return names_bat


def export_file_names(resized_df):
    filename_queue = list(resized_df['Full paths'])
    with open(resized_root + '/' + 'file_name_queue.txt', 'w') as file:
        for i in filename_queue:
            file.write(i + '\n')


def main():
    resized_df = pd.read_csv('image_databases/resized_logos_df.csv', index_col=0)
    # analyze_band_names(resized_df)
    resized_df = clean_df_names(resized_df)
    resized_df.to_csv('image_databases/resized_logos_df.csv')
    # analyze_cleaned_names(resized_df)
    names_bat = build_name_matrices(resized_df)
    np.save(resized_root + '/' + 'name_matrices', names_bat)
    export_file_names(resized_df)
    save_tf_binaries(resized_df, names_bat, 256)
    # save_tf_binaries(resized_df, names_bat, 512)
    # save_tf_binaries(resized_df, names_bat, 128)
    # save_tf_binaries(resized_df, names_bat, 64)
    # save_tf_binaries(resized_df, names_bat, 32)



def save_tf_binaries(resized_df, names_bat, img_size):
    print('Building binary for ' + str(img_size) + '...')
    example_IDs = list(resized_df.index)

    np.random.shuffle(example_IDs)
    with tf.python_io.TFRecordWriter(resized_root + '/' + str(img_size) + '_images_and_names.tfrecords') as writer:
        count = 0
        for example_num in tqdm(example_IDs):
            count += 1
            if count > 128*3:
                break
            img_file_name = resized_df['Full paths'][example_num]
            image = np.array(misc.imread(resized_root + '/' + str(img_size) + '/' + img_file_name))
            image = np.reshape(image, (img_size ** 2))
            label_matrix = names_bat[example_num]
            label_matrix = np.reshape(label_matrix, (20*39))

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'label_matrix': tf.train.Feature(
                            float_list=tf.train.FloatList(value=label_matrix.astype('float'))
                            ),
                        'image': tf.train.Feature(
                            float_list=tf.train.FloatList(value=image.astype('float'))
                            )
                        }
                )
            )

            serialized_data = example.SerializeToString()
            writer.write(serialized_data)
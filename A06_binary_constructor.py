import pandas as pd
import csv
import numpy as np
from utils.greyscaling import plot_arbitrary_array
from utils.utils import progress_bar



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
    return names_bat


def main():
    resized_df = pd.read_csv('image_databases/resized_logos_df.csv', index_col=0)
    analyze_band_names(resized_df)
    resized_df = clean_df_names(resized_df)
    resized_df.to_csv('image_databases/resized_logos_df.csv')
    analyze_cleaned_names(resized_df)
    names_bat = build_name_matrices(resized_df)
    np.save('name_matrices', names_bat)


filename_queue = list(resized_df['Full paths'])
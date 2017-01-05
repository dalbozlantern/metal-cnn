import os
from utils.utils import progress_bar
import pandas as pd
import math

# Load project variables
from configparser import ConfigParser
config = ConfigParser()
config.read('config.ini')
img_root = config.get('main', 'img_root')

# =========================================================================

def move_to_bins(bands_df, bin_size=1000):

    # Initialize
    number_of_entries = bands_df.shape[0]
    folder_bins = []
    full_paths = []
    with open('file_move_log.ini', 'a') as file:
        file.write('STARTING NEW RUN****************\n')

    # Iterate over all images
    for count in range(number_of_entries):
        if count % 50 == 0:
            progress_bar('Processing image #', count, number_of_entries + 1)

        bucket_number = bin_size * math.floor(count / bin_size)
        formatted_bucket = '{0:0>4}'.format(bucket_number)
        folder_bins += [formatted_bucket]
        file_name = bands_df['Logo file'][count]
        full_paths += [str(formatted_bucket) + '/' + str(file_name)]

        if not os.path.isdir(img_root + '/' + formatted_bucket):
            os.makedirs(img_root + '/' + formatted_bucket)

        try:
            old_file_path = img_root + '/' + str(file_name)
            new_filename = img_root + '/' + str(formatted_bucket) + '/' + str(file_name)
            if os.path.isfile(old_file_path) and file_name != 'NO LOGO FOUND':
                os.rename(old_file_path, new_filename)
        except:
            with open('file_move_log.ini', 'a') as file:
                file.write('Error on ' + old_file_path + '\n')

    return folder_bins, full_paths


def main():
    bands_df = pd.read_csv('scraping/bands_df.csv', index_col=0)
    folder_bins, full_paths = move_to_bins(bands_df)
    bands_df['Bins'] = folder_bins
    bands_df['Full paths'] = full_paths
    bands_df.to_csv('scraping/bands_df.csv')
    print('\nDone.')
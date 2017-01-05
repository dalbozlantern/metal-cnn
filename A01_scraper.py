# =========================================================================
# Description: This file identifies and pulls the images off of metal-archives
# =========================================================================


import urllib
import os
import pandas as pd
import json
from bs4 import BeautifulSoup
import time
from utils.utils import progress_bar

# Load project variables
from configparser import ConfigParser
config = ConfigParser()
config.read('config.ini')
img_root = config.get('main', 'img_root')

# =========================================================================

crawl_delay = 3  # As per the metal-archives robots.txt


def download_json_files():

    # Downloads the .json files from metal-archives.com that contain the urls for all band names in the database

    def initialize_json_url_list():
        key_list = []
        for letter_number in range(0, 26):
            key_list += [chr(65 + letter_number)]
        key_list += ['NBR', '~']
        return key_list

    def compile_json_url(letter_lookup, data_index):
        json_url_to_download = 'http://www.metal-archives.com/browse/ajax-letter/l/' + letter_lookup + '/json/1?s' \
                                                                                                       'Echo=2' \
                                                                                                       '&iColumns=4' \
                                                                                                       '&sColumns=' \
                                                                                                       '&iDisplayStart=' + str(
            500 * data_index) + \
                               '&iDisplayLength=500' \
                               '&mDataProp_0=0' \
                               '&mDataProp_1=1' \
                               '&mDataProp_2=2' \
                               '&mDataProp_3=3' \
                               '&iSortCol_0=0' \
                               '&sSortDir_0=asc' \
                               '&iSortingCols=1' \
                               '&bSortable_0=true' \
                               '&bSortable_1=true' \
                               '&bSortable_2=true' \
                               '&bSortable_3=false' \
                               '&_=1481858180439'
        return json_url_to_download

    #Initialize url iteration
    print('Initializing json download:')
    letter_indeces = initialize_json_url_list()
    json_counter = 0
    for letter_lookup in letter_indeces:
        minimum_json_lines = 8
        for data_index in range(0, 100):
            print('    Processing ' + letter_lookup + ', #' + str(data_index) + ' (total saved: ' + str(json_counter) + ')...', end='\r')

            # Download the file
            json_url_to_download = compile_json_url(letter_lookup, data_index)
            response = urllib.request.urlopen(json_url_to_download)
            time.sleep(crawl_delay)
            json_content = response.read().decode('UTF-8')

            # Break to next letter if file has no entries
            json_lines_returned = len(json_content.split('\n'))
            if json_lines_returned < minimum_json_lines:
                break

            # Save to drive
            json_counter += 1
            save_file_name = 'scraping/01-json/' + letter_lookup + '-' + str(data_index) + '.json'
            with open(save_file_name, 'w') as json_file:
                json_file.write(json_content)

    print('\n    Done downloading json files, total = ' + str(json_counter))
    return


def parse_downloaded_json_files():

    # Searches through the json files that were downloaded and pulls the band webpages into a csv

    print('Parsing downloaded json files:')
    band_links = []
    band_names = []
    band_countries = []
    band_genres = []

    # Iterate through each file
    file_counter = 0
    for root, dirs, filenames in os.walk('scraping/01-json'):
        for file_name in filenames:
            if file_counter % 5 == 0:
                print('    Scanning file #' + str(file_counter) + '...', end='\r')
            file_counter += 1
            save_file_name = os.path.join(root, file_name)
            with open(save_file_name, 'r') as json_file:
                json_string = json_file.read()
            parsed_json = json.loads(json_string)
            bands_data = parsed_json['aaData']
            for band_entry in bands_data:
                soup = BeautifulSoup(band_entry[0], 'html.parser')
                link = soup.find_all('a')[0]

                band_links += [link['href']]
                band_names += link.contents
                band_countries += [band_entry[1]]
                band_genres += [band_entry[2]]

    bands_black = ['ebm' in i or 'black' in i for i in band_genres]
    # Create and export a dataframe for the bands
    bands_df = pd.DataFrame({
        'Band': band_names,
        'Url': band_links,
        'Genre': band_genres,
        'Country': band_countries,
        'Black': bands_black,
    },)
    bands_df['Logo url'] = None
    bands_df['Logo file'] = None
    bands_df.to_csv('image_databases/downloaded_bands_df.csv')

    number_of_bands = len(band_names)
    print('\n    Done scanning json files: ' + str(number_of_bands) + ' band names found')

    return bands_df


def parse_band_urls(bands_df, iteration_limit=0):

    # Downloads the images across a range of bands

    def download_image_for_band(band_index, bands_df):
        # Parse the band's page
        test_url = bands_df['Url'][band_index]
        band_webpage = urllib.request.urlopen(test_url)
        time.sleep(crawl_delay)
        band_web_content = band_webpage.read().decode('UTF-8')
        soup = BeautifulSoup(band_web_content, 'html.parser')

        # Find the link to the logo
        try:
            logo_link = soup.find_all(id='logo')[0]['href']
        except:
            bands_df.loc[band_index, 'Logo url'] = 'NO LOGO FOUND'
            bands_df.loc[band_index, 'Logo file'] = 'NO LOGO FOUND'
            return bands_df

        bands_df.loc[band_index, 'Logo url'] = logo_link

        # Save the image to disk
        logo_ext_pos = str.rfind(logo_link, '.')
        logo_extension = logo_link[logo_ext_pos:]
        logo_file = bands_df['Band'][band_index][0:10] + '-#' + str(band_index) + logo_extension
        urllib.request.urlretrieve(logo_link, img_root + logo_file)
        time.sleep(crawl_delay)
        bands_df.loc[band_index, 'Logo file'] = logo_file

        return bands_df

    print('Identifying and downloading images')
    if iteration_limit == 0:
        iteration_limit = bands_df.shape[0]
    open('scraping/band_url_parser_log.ini', 'w').write('Band url parsing log:\n\n')

    for band_index in range(0, iteration_limit):
        if pd.isnull(bands_df['Logo url'][band_index]):
            progress_bar('Band #', band_index, iteration_limit)
            try:
                bands_df = download_image_for_band(band_index, bands_df)
            except:
                open('scraping/band_url_parser_log.ini', 'a').write('Error with ' + str(band_index) + '\n')
            if band_index % 5 == 0 or band_index == iteration_limit:
                bands_df.to_csv('image_databases/downloaded_bands_df.csv')

    print('\n    Done parsing band URLs')
    return bands_df


# =========================================================================


def main():
    print('Rebuilding the ENTIRE database from scratch in 30 SEC...')
    time.sleep(30)
    download_json_files()
    bands_df = parse_downloaded_json_files()
        # bands_df = pd.read_csv('image_databases/downloaded_bands_df.csv', index_col=0)
    bands_df = parse_band_urls(bands_df)
    print('Finished all steps.')
    return

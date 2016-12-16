import urllib
import os
import pandas as pd
from bs4 import BeautifulSoup


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
    print('Parsing downloaded json files:')
    band_links = []
    band_names = []

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
            soup = BeautifulSoup(json_string, 'html.parser')
            links = soup.find_all('a')

            for link in links:
                band_links += [link['href']]
                band_names += link.contents

    number_of_bands = len(band_names)
    print('\n    Done scanning json files: ' + str(number_of_bands) + ' band names found')

    bands_df = pd.DataFrame({'Band': band_names, 'Url': band_links})
    return bands_df

bands_df = parse_downloaded_json_files()

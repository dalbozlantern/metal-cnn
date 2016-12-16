import urllib.request
import json
from bs4 import BeautifulSoup


def initialize_json_url_list():
    key_list = []
    for letter_number in range(0, 26):
        key_list += [chr(65 + letter_number)]
    key_list += ['NBR', '~']
    return key_list



def download_json_files():

    def compile_json_url(letter_lookup, data_index):
        json_url_to_download = 'http://www.metal-archives.com/browse/ajax-letter/l/' + letter_lookup + '/json/1?s' \
                                'Echo=2' \
                                '&iColumns=4' \
                                '&sColumns=' \
                                '&iDisplayStart=' + str(500 * data_index) + \
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

    print('\nDone downloading json files, total = ' + str(json_counter))



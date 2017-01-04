import pandas as pd
import csv
import re
from random import randrange


def whitelist_cleaning(bands_df):

    print('Screening band names for non-latin unicode characters...')

    # Initialize
    band_names = list(bands_df['Band'])
    # Import the list of find-and-replace accent marks
    accent_dict = {rows[0]: rows[1] for rows in
                   csv.reader(open('utils/latin_accents.csv', mode='r'))}

    # Cleans the band names, returns a fidelity score
    def whitelist_unicode(content, accent_dict):
        # Downgrade accents to regular latin characters
        for accented_letter, accented_letter_replacement in accent_dict.items():
            content = content.replace(accented_letter, accented_letter_replacement)

        # Removes all non-white_list_unicoded unicode characters and measures the level of 'fixing'
        inital_length = len(content)
        content = re.sub(r'[^a-zA-Z0-9 \n\\\'\"!#*$%\(\),\-\/?:;\.…&+><]', '', content)
        abs_score = len(content) - inital_length

        try:
            clean_score = 1 + abs_score / inital_length
        except:
            clean_score = 1

        return content, clean_score

    # Returns the cleaned unicode band names and fidelity scores
    def unicode_cleaner(band_names):
        # Iterate over band names
        whitelist_scores = []
        cleaned_names = []
        for name in band_names:
            clean_name, score = whitelist_unicode(name, accent_dict)
            whitelist_scores += [score]
            cleaned_names += [clean_name]
        return cleaned_names, whitelist_scores

    cleaned_names, whitelist_scores = unicode_cleaner(band_names)

    list_if_valid = [score == 1 for score in whitelist_scores]

    bands_df['Unicode valid'] = list_if_valid
    bands_df['Cleaned name'] = cleaned_names
    return bands_df


def print_statistics(bands_df):

    has_image = sum(bands_df['Logo file'] != 'NO LOGO FOUND')
    img_percent = has_image / bands_df.shape[0]
    img_percent = round(100 * img_percent, 1)
    print(str(img_percent) + '% of bands have an valid logo')

    uni_percent = sum(bands_df['Unicode valid']) / bands_df.shape[0]
    uni_percent = round(100 * uni_percent, 1)
    print(str(uni_percent) + '% of band names pass unicode screening')

    all_percent = sum(bands_df['Overall valid']) / bands_df.shape[0]
    all_percent = round(100 * all_percent, 1)
    print(str(all_percent) + '% of band names pass both criteria')


def print_a_few(input_list, spacer='    '):
    for i in range(5):
        print(spacer + input_list[randrange(len(input_list))])


def show_examples(bands_df):
    band_names = list(bands_df['Band'])
    valid_list = list(bands_df['Unicode valid'])
    cleaned_names = list(bands_df['Cleaned name'])

    no_change = [band_names[i] for i in range(len(band_names))
                 if valid_list[i] == 1]
    changed_but_valid = [band_names[i] + ' --> ' + cleaned_names[i] for i in range(len(band_names))
                         if valid_list[i] == 1
                         and cleaned_names[i] != band_names[i]]
    invalid_names = [band_names[i] for i in range(len(band_names))
                         if valid_list[i] == 0]

    print('\nSome band names that didn\'t need unicode cleaning:')
    print_a_few(no_change)
    print('\nSome band names that made the cut, but needed cleaning:')
    print_a_few(changed_but_valid)
    print('\nSome band names that were cut:')
    print_a_few(invalid_names)


def main():
    bands_df = pd.read_csv('scraping/bands_df.csv')
    bands_df = whitelist_cleaning(bands_df)
    bands_df['Overall valid'] = (bands_df['Logo file'] != 'NO LOGO FOUND') & bands_df['Unicode valid']
    bands_df.to_csv('scraping/bands_df.csv')
    print_statistics(bands_df)
    show_examples(bands_df)
    return bands_df


bands_df = main()
# Only pass through whitelisted unicode characters

import pandas as pd
import csv
import re
import math

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

    percent_valid = round(sum(list_if_valid)/len(list_if_valid),2)
    print(str(100*percent_valid) + '% of band names pass unicode screening')

    bands_df['Unicode valid'] = list_if_valid
    bands_df['Cleaned name'] = cleaned_names
    return bands_df


bands_df = pd.read_csv('scraping/bands_df.csv')
bands_df = whitelist_cleaning(bands_df)

band_names = list(bands_df['Band'])
valid_list = list(bands_df['Unicode valid'])
cleaned_names = list(bands_df['Cleaned name'])

no_change = [band_names[i] + ' | ' + cleaned_names[i] for i in range(len(band_names))
             if valid_list[i] == 1]
changed_but_valid = [band_names[i] + ' | ' + cleaned_names[i] for i in range(len(band_names))
                     if valid_list[i] == 1
                     and cleaned_names[i] != band_names[i]]
invalid_names = [band_names[i] + ' | ' + cleaned_names[i] for i in range(len(band_names))
                     if valid_list[i] == 0]
print('*******')
for i in range(10):
    print(no_change[i])
print('*******')
for i in range(10):
    print(changed_but_valid[i])
print('*******')
for i in range(10):
    print(invalid_names[i])


excluded_strings = []
for i in range(len(band_names)):
    excluded_strings += list(set(band_names[i]) - set(cleaned_names[i]))
with open('test.txt', 'w') as file:
    for i in set(excluded_strings):
        file.write(i + '\n')

logo_files = bands_df['Logo file']
test =
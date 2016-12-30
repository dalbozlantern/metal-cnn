import pandas as pd
import copy
import re

# Genre_analysis
def genre_analysis(band_genres, band_names):
    band_genres_dupe = copy.deepcopy(band_genres)
    band_genres_dupe = [genre.lower() for genre in band_genres_dupe]
    band_genres_dupe = [str.replace(genre, '  ', ' ') for genre in band_genres_dupe]
    band_genres_dupe = [str.replace(genre, ', ', '/') for genre in band_genres_dupe]
    band_genres_dupe = [str.replace(genre, ' and ', '/') for genre in band_genres_dupe]
    band_genres_dupe = [str.replace(genre, ' metal', '') for genre in band_genres_dupe]
    band_genres_dupe = [str.replace(genre, ' elements', ' influences') for genre in band_genres_dupe]
    band_genres_dupe = [re.sub(r'\([a-zA-Z0-9,/\-\' ;"\.&]*\)', '', genre) for genre in band_genres_dupe]
    band_genres_dupe = [str.replace(genre, ' (early', '') for genre in band_genres_dupe]

    bands_database = {}
    for entry_index in range(0, len(band_genres)):
        bands_database[entry_index] = {
            'Raw': band_genres[entry_index],
            'Band': band_names[entry_index],
            'Formatted': band_genres_dupe[entry_index],
        }
    for entry_index in bands_database:
        entry = bands_database[entry_index]
        formatted = entry['Formatted']
        try:
            search_term = re.findall(r'([\s\S]+) with ([\s\S]+) influences', formatted)[0]
            bands_database[entry_index]['Genres'] = search_term[0].split('/')
            bands_database[entry_index]['Influences'] = search_term[1].split('/')
        except:
            bands_database[entry_index]['Genres'] = formatted.split('/')
            bands_database[entry_index]['Influences'] = []

    return bands_database


def list_overall_genres(bands_database):
    overall_genres = []
    for entry_index in bands_database:
        entry = bands_database[entry_index]
        overall_genres += entry['Genres']
    overall_genres = set(overall_genres)
    return overall_genres



def merge_dfs(bands_df, bands_df_dupe):
    a = pd.merge(bands_df_dupe, bands_df, on='Url')
    for v in {'Unnamed: 0',
              'Unnamed: 0.1',
              'Unnamed: 0.1.1',
              'Band_y',
              'Genre_y',
              'Country_y'
              }: del a[v]
    a.columns = ['Band', 'Country', 'Genre', 'Url', 'Black', 'Logo url', 'Logo file']
    return

import pandas as pd
import matplotlib.pyplot as plt

bands_df = pd.read_csv('scraping/bands_df.csv')
band_names = list(bands_df['Band'])
lengths = [len(name) for name in band_names]
lengths.sort()
indices = [i/len(lengths) for i in list(range(0, len(lengths)))]

plt.plot(indices, lengths)
plt.yscale('log')
plt.show()

import collections
a = collections.Counter('')
for i in band_names:
    a += collections.Counter(i)
b = dict(a)
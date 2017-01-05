from utils.greyscaling import *
from utils.utils import progress_bar

# Load project variables
from configparser import ConfigParser
config = ConfigParser()
config.read('config.ini')
img_root = config.get('main', 'img_root')


def analyze_image_scores(bands_df):
    corpus_size = bands_df.shape[0]
    with open('scores_log.ini', 'a') as file:
        file.write('NEW RUN*********')
    for i, row in bands_df.iterrows():
        if i % 50 == 0:
            progress_bar('Processing image #', i, corpus_size)
        try:
            if (bands_df['Blur scores'][i] == None or \
                            bands_df['Border scores'][i] == None) & \
                            bands_df['Overall valid'][i]:
                file_name = img_root + '/' + bands_df['Full paths'][i]
                image = load_image(file_name)
                greyscale_image = max_rgb2grey(image)
                normalized_image = normalize_image(greyscale_image)
                border_score = find_border_score(normalized_image)
                blur_score = find_blur_score(normalized_image)
                bands_df.set_value(i, 'Blur scores', blur_score)
                bands_df.set_value(i, 'Border scores', border_score)
        except:
            with open('scores_log.ini', 'a') as file:
                file.write(bands_df['Full paths'][i] + '\n')
    return bands_df


def validate_scores(bands_df, blur_threshold=.025, border_threshold=.01):
    corpus_size = bands_df.shape[0]
    with open('validation_log.ini', 'a') as file:
        file.write('NEW RUN*********')
    for i, row in bands_df.iterrows():
        if i % 50 == 0:
            progress_bar('Processing image #', i, corpus_size)
        try:
            if bands_df['Unicode valid'][i] and \
                            bands_df['Logo file'][i] != 'nan' and \
                            bands_df['Logo file'][i] != 'NO LOGO FOUND':
                if bands_df['Border scores'][i] <= border_threshold:
                    bands_df.set_value(i, 'Border valid', True)
                else:
                    bands_df.set_value(i, 'Border valid', False)
                if bands_df['Blur scores'][i] <= blur_threshold:
                    bands_df.set_value(i, 'Blur valid', True)
                else:
                    bands_df.set_value(i, 'Blur valid', False)
                if bands_df['Blur valid'][i] and bands_df['Border valid'][i]:
                    bands_df.set_value(i, 'Overall valid', True)
        except:
            bands_df.set_value(i, 'Overall valid', False)
            with open('scores_log.ini', 'a') as file:
                file.write(bands_df['Full paths'][i] + '\n')
    return bands_df


def view_distributions(bands_df, key, number_of_examples=20, output_file='test.html'):
    bands_df_copy = bands_df.dropna()
    bands_df_copy = bands_df_copy.sort(key)
    file_list = [img_root + '/' + i for i in bands_df_copy['Full paths']]
    score_list = list(bands_df_copy[key])
    # file_list = [img_root + '/' + bands_df_copy['Full paths'][i]
    #              for i in range(bands_df_copy.shape[0])
    #              if bands_df_copy[key][i] is not None]
    # score_list = [i for i in bands_df_copy[key]
    #               if i is not None]
    assert len(file_list) == len(score_list)
    save_to_html(output_file, file_list, score_list, number_of_examples)


def score_statistics(bands_df):
    bands_df_copy = bands_df.dropna()
    blur_scores = [i for i in bands_df_copy['Blur scores'] if i is not None]
    border_scores = [i for i in bands_df_copy['Border scores'] if i is not None]
    blur_scores.sort()
    border_scores.sort()
    plot_arbitrary_array(blur_scores)
    plot_arbitrary_array(border_scores)
    print('% of images above the blur score threshold')
    print(format_as_percent(np.mean(bands_df['Blur valid'])))
    print('# of images above the blur score threshold')
    print(np.sum(bands_df['Blur valid']))
    print('% of images above the border score threshold')
    print(format_as_percent(np.mean(bands_df['Border valid'])))
    print('# of images above the border score threshold')
    print(np.sum(bands_df['Border valid']))
    print('% of validated images in the corpus (including border, blur, and unicode)')
    print(format_as_percent(np.mean(bands_df['Overall valid'])))
    print('# of validated images in the corpus (including border, blur, and unicode)')
    print(np.sum(bands_df['Overall valid']))


def main():
    bands_df = pd.read_csv('image_databases/downloaded_bands_df.csv', index_col=0)

    bands_df['Border scores'] = None
    bands_df['Blur scores'] = None
    bands_df = analyze_image_scores(bands_df)
    bands_df.to_csv('image_databases/downloaded_bands_df.csv')

    bands_df['Border valid'] = None
    bands_df['Blur valid'] = None
    bands_df['Overall valid'] = False
    bands_df = validate_scores(bands_df)
    bands_df.to_csv('image_databases/downloaded_bands_df.csv')

    view_distributions(bands_df, 'Blur scores', 100, 'blur_scores.html')
    view_distributions(bands_df, 'Border scores', 100, 'border_scores.html')
    score_statistics(bands_df)





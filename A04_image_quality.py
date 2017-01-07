from utils.greyscaling import *
from utils.image_splitting import *
# (inherits all imports)

from utils.utils import progress_bar
import re

# Load project variables
from configparser import ConfigParser
config = ConfigParser()
config.read('config.ini')
img_root = config.get('main', 'img_root')
cropped_root = config.get('main', 'cropped_root')
resized_root = config.get('main', 'resized_root')


def analyze_image_scores(bands_df):
    corpus_size = bands_df.shape[0]
    with open('scores_log.ini', 'a') as file:
        file.write('NEW RUN*********\n')
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
        file.write('NEW RUN*********\n')
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


def initialize_cropped_df():
    cropped_df = pd.DataFrame(columns=['Band',
                               'Country',
                               'Genre',
                               'Black',
                               'Full paths',
                               'Height',
                               'Width',
                               'Diag',
                               'Valid size',
                               'Number extracted',
                               ])
    return cropped_df


def export_cropped_images(bands_df, cropped_df):
    with open('cropping_log.ini', 'a') as file:
        file.write('NEW RUN*********\n')

    corpus_size = bands_df.shape[0]
    for i, row in bands_df.iterrows():
        if i % 100 == 0:
            progress_bar('Processing image #', i, corpus_size)
            cropped_df.to_csv('image_databases/cropped_bands_df.csv')
        try:
            if bands_df['Overall valid'][i] and (not (cropped_df['Full paths'].fillna('missing') == bands_df['Full paths'][i]).any()):

                formatted_bucket = '{0:0>4}'.format(bands_df['Bins'][i])
                if not os.path.isdir(cropped_root + '/' + formatted_bucket):
                    os.makedirs(cropped_root + '/' + formatted_bucket)

                file_name = img_root + '/' + bands_df['Full paths'][i]
                image = load_image(file_name)
                greyscale_image = max_rgb2grey(image)
                normalized_image = normalize_image(greyscale_image)
                cropped_image = remove_bounding_box(normalized_image)
                extracted_masks = mark_images_for_splitting(normalized_image)
                number_extracted = len(extracted_masks)
                carryover_data = ['Band', 'Country', 'Genre', 'Black']

                height = normalized_image.shape[0]
                width = normalized_image.shape[1]
                diag = math.sqrt(height**2 + width**2)

                new_name = bands_df['Full paths'][i]
                new_name = re.sub(r'\?\d+', '', new_name)
                save_image(cropped_image, cropped_root, new_name)

                new_data_row = {'Valid size': None}
                for entry in carryover_data:
                    new_data_row[entry] = bands_df[entry][i]
                new_data_row['Height'] = height
                new_data_row['Width'] = width
                new_data_row['Diag'] = diag
                new_data_row['Number extracted'] = number_extracted
                new_data_row['Full paths'] = new_name
                cropped_df = cropped_df.append(new_data_row, ignore_index=True)
        except:
            with open('cropping_log.ini', 'a') as file:
                file.write(bands_df['Full paths'][i] + '\n')
    return cropped_df


def move_splits_for_manual_review(cropped_df):

    duplicates_df = cropped_df.loc[cropped_df['Number extracted'] > 1]
    duplicates_size = duplicates_df.shape[0]
    duplicates_df.index = range(duplicates_size)

    for i, row in duplicates_df.iterrows():
        if i % 50 == 0:
            progress_bar('Processing image #', i, duplicates_size)

        file_name = cropped_root + '/' + duplicates_df['Full paths'][i]
        image = load_image(file_name)
        greyscale_image = max_rgb2grey(image)
        normalized_image = normalize_image(greyscale_image)

        formatted_bucket = duplicates_df['Full paths'][i]
        if formatted_bucket[0] == '/':
            formatted_bucket = formatted_bucket[1:]
        formatted_bucket = re.sub(r'/[\s\S]*', '', duplicates_df['Full paths'][i])
        if not os.path.isdir(cropped_root + '/dupes/' + formatted_bucket):
            os.makedirs(cropped_root + '/dupes/' + formatted_bucket)

        save_image(individual_image, cropped_root + '/dupes/', duplicates_df['Full paths'][i])


def re_import_dupes(cropped_df):
    master_count = -1
    for root, dirs, filenames in os.walk(cropped_root + '/dupes'):
        for file_name in filenames:
            master_count += 1

    count = -1
    for root, dirs, filenames in os.walk(cropped_root + '/dupes'):
        for file_name in filenames:
            count += 1

            if count % 50 == 0:
                progress_bar('Processing image #', count, master_count)

            full_path = os.path.join(root, file_name)
            bin_path = full_path[full_path[:full_path.rfind('/')].rfind(
                '/') + 1:]  # Returns onward from after the second '/' from the right
            dataframe_row = cropped_df[cropped_df['Full paths'] == bin_path]
            dataframe_index = dataframe_row.index.tolist()[0]
            cropped_data = dict(cropped_df.ix[dataframe_index])

            image = load_image(full_path)
            greyscale_image = max_rgb2grey(image)
            normalized_image = normalize_image(greyscale_image)
            extracted_images = return_split_and_cropped_images(normalized_image)

            try:
                os.remove(cropped_root + '/' + bin_path)
            except:
                pass
            cropped_df.drop(cropped_df.index[[1, 3]])

            for image_num in extracted_images:
                individual_image = extracted_images[image_num]
                height = individual_image.shape[0]
                width = individual_image.shape[1]
                diag = math.sqrt(height ** 2 + width ** 2)
                new_name = bin_path
                if new_name[0] == '/':
                    new_name = new_name[1:]
                if image_num >= 2:
                    new_name = new_name[:-4] + '_' + str(image_num) + new_name[-4:]
                cropped_data['Full paths'] = new_name
                cropped_data['Height'] = height
                cropped_data['Width'] = width
                cropped_data['Diag'] = diag

                save_image(individual_image, cropped_root, new_name)
                cropped_df = cropped_df.append(cropped_data, ignore_index=True)
    return cropped_df


def image_size_analysis(cropped_df):
    # Generate aspect ratios
    cropped_df['Aspect ratios'] = np.divide(cropped_df['Width'], cropped_df['Height'])
    log_ars = np.log(cropped_df['Aspect ratios'])
    cropped_df['Intuitive aspect ratios'] = np.multiply(np.sign(log_ars), np.exp(np.abs(log_ars)))
    cropped_df['Abs aspect ratios'] = np.exp(np.abs(log_ars))

    # Analyze aspect ratios
    int_aspect_ratios = list(cropped_df['Intuitive aspect ratios'])
    int_aspect_ratios.sort()
    plot_arbitrary_array(int_aspect_ratios, [-2, 10])

    # Trim based on aspect ratios
    upper_threshold = 6
    lower_threshold = -1.5
    cropped_df['Valid size'] = np.multiply(cropped_df['Abs aspect ratios'] >= lower_threshold,
                                           cropped_df['Abs aspect ratios'] <= upper_threshold)
    percent_saved = np.mean(cropped_df['Valid size'])
    print(percent_saved)

    # Analyze resizing
    target_box = 512
    upscaling_threshold = 3
    cropped_df['Max dim'] = np.maximum(cropped_df['Height'], cropped_df['Width'])
    cropped_df['Scale factor'] = np.divide(target_box, cropped_df['Max dim'])
    scale_factor = list(cropped_df['Scale factor'])
    scale_factor.sort()
    plot_arbitrary_array(scale_factor)

    # Trim based on resizing
    cropped_df['Valid size'] = np.multiply(np.multiply(cropped_df['Abs aspect ratios'] >= lower_threshold,
                                                       cropped_df['Abs aspect ratios'] <= upper_threshold),
                                           cropped_df['Scale factor'] <= upscaling_threshold)
    percent_saved = np.mean(cropped_df['Valid size'])
    print(percent_saved)
    return cropped_df


def pad_to_square_black(normalized_image):
    max_dim = max(normalized_image.shape)
    min_dim = min(normalized_image.shape)
    short_axis = normalized_image.shape.index(min_dim)
    short_padding = math.floor((max_dim - min_dim) / 2)
    long_padding = math.ceil((max_dim - min_dim) / 2)
    short_indices = [short_padding, max_dim]
    long_indices = [long_padding, max_dim]
    short_pad = np.zeros((short_indices[short_axis], short_indices[1-short_axis]))
    long_pad = np.zeros((long_indices[short_axis], long_indices[1-short_axis]))
    new_image = np.concatenate((short_pad, normalized_image, long_pad), short_axis)
    new_shape = new_image.shape
    assert new_shape[0] == new_shape[1]
    return new_image


def initialize_resized_df():
    cropped_df = pd.DataFrame(columns=['Band',
                               'Country',
                               'Genre',
                               'Black',
                               'Full paths',
                               ])
    return cropped_df


def resize_library(cropped_df, resized_df):
    with open('resizing_log.ini', 'a') as file:
        file.write('NEW RUN*********\n')

    carryover_data = ['Band', 'Country', 'Genre', 'Black']
    corpus_size = cropped_df.shape[0]

    for i, row in cropped_df.iterrows():
        if i % 100 == 0:
            progress_bar('Processing image #', i, corpus_size)
            resized_df.to_csv('image_databases/resized_logos_df.csv')
        try:
            if cropped_df['Valid size'][i]:

                old_name = cropped_df['Full paths'][i]

                formatted_bucket = old_name
                if formatted_bucket[0] == '/':
                    formatted_bucket = formatted_bucket[1:]
                formatted_bucket = re.sub(r'/[\s\S]*', '', old_name)
                if not os.path.isdir(resized_root + '/' + formatted_bucket):
                    os.makedirs(resized_root + '/' + formatted_bucket)

                file_name = cropped_root + '/' + old_name
                image = load_image(file_name)
                normalized_image = max_rgb2grey(image)
                normalized_image = normalized_image
                padded_image = pad_to_square_black(normalized_image)
                loaded_image = PIL.Image.fromarray(padded_image)
                resized_image = loaded_image.resize([512, 512], PIL.Image.ANTIALIAS)

                new_name = old_name[:old_name.rfind('.')] + '.png'
                save_image(np.asarray(resized_image), resized_root, new_name)

                new_data_row = {'Full paths': new_name}
                for entry in carryover_data:
                    new_data_row[entry] = cropped_df[entry][i]
                resized_df = resized_df.append(new_data_row, ignore_index=True)
        except:
            with open('resizing_log.ini', 'a') as file:
                file.write(bands_df['Full paths'][i] + '\n')

    return resized_df


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

    view_distributions(bands_df, 'Blur scores', 100, 'outputs/blur_scores.html')
    view_distributions(bands_df, 'Border scores', 100, 'outputs/border_scores.html')
    score_statistics(bands_df)

    cropped_df = initialize_cropped_df()
    cropped_df = export_cropped_images(bands_df, cropped_df)
    cropped_df.to_csv('image_databases/cropped_bands_df.csv')

    move_splits_for_manual_review(cropped_df)
    # ++++++++++++++++++++++++++++++++++++++++
    # BREAK
    # Then manually pruned false positives, which was most of them
    # ++++++++++++++++++++++++++++++++++++++++
    cropped_df = re_import_dupes(cropped_df)
    cropped_df.to_csv('image_databases/cropped_bands_df.csv')

    cropped_df = image_size_analysis(cropped_df)

    resized_df = initialize_resized_df()
    resized_df = resize_library(cropped_df, resized_df)
    resized_df.to_csv('image_databases/resized_logos_df.csv')

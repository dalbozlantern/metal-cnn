import numpy as np
from utils.greyscaling import remove_bounding_box

def scale_values(binary_array):
    return 1 - 2 * np.array(binary_array)


def un_scale_values(binary_array):
    return 1 - 2 * np.array(binary_array)


def iterate_flipping_list(sequence_lengths, scaled_values, threshold=5):
    valid_chunks = np.greater_equal(sequence_lengths, threshold)
    expanded_comparison = [False] + list(valid_chunks) + [False]
    adjacent_to_valid = [(expanded_comparison[i-1] or expanded_comparison[i+1])
                         and not expanded_comparison[i]
                         for i in range(1, len(expanded_comparison) - 1)]
    inversion_array = scale_values(adjacent_to_valid)
    flipped_values = np.multiply(scaled_values, inversion_array)
    return flipped_values


def downshift_array(sequence_lengths, scaled_values):
    cum_sums = [sequence_lengths[0]]
    for i in range(1, len(scaled_values)):
        if scaled_values[i] == scaled_values[i-1]:
            cum_sums += [sequence_lengths[i] + cum_sums[i-1]]
        else:
            cum_sums += [sequence_lengths[i]]
    right_to_left_unique = [scaled_values[i] != scaled_values[i+1] for i in range(len(scaled_values) - 1)] + [True]
    sequence_lengths = [cum_sums[i] for i in range(len(right_to_left_unique)) if right_to_left_unique[i]]
    scaled_values = [scaled_values[i] for i in range(len(right_to_left_unique)) if right_to_left_unique[i]]
    return sequence_lengths, scaled_values


def collapse_array(sequence_lengths, scaled_values):
    for i in range(1000):
        assert i < 999
        flipped_values = iterate_flipping_list(sequence_lengths, scaled_values)
        prior_len = len(scaled_values)
        sequence_lengths, scaled_values = downshift_array(sequence_lengths, flipped_values)
        if len(scaled_values) == prior_len:
            break
    return sequence_lengths, scaled_values


def extract_preliminary_sequence_lengths(input_array):
    # The input array toggles between 0 and 1; so, find out where the array changes
    change_from_previous = np.array(
        [1] + [input_array[i] != input_array[i - 1] for i in range(1, len(input_array))])
    change_positions = [i for i in range(len(change_from_previous)) if change_from_previous[i] == 1]
    change_positions += [len(
        change_from_previous)]  # Tack on the overall length of the array to the end for subsequent manipulations

    # Now build a new array that just contains the lengths of the sequences of 0's and 1's
    sequence_lengths = np.array(
        [change_positions[i] - change_positions[i - 1] for i in range(1, len(change_positions))]
    )
    scaled_values = [2 * input_array[i] - 1 for i in change_positions[:len(change_positions) - 1]]
    return sequence_lengths, scaled_values


def extract_bookmarks(sequence_lengths, scaled_values):
    all_bookmarks = [0] + list(np.cumsum(sequence_lengths))
    left_bookends = all_bookmarks[:-1]
    right_bookends = all_bookmarks[1:]
    masks = [(left_bookends[i], right_bookends[i]) for i in range(len(scaled_values)) if scaled_values[i] == -1]
    return masks


def extract_vertical_clipping_masks(normalized_image, blackness_threshold=2.5):
    avg_inten = np.mean(normalized_image, 1)
    pixels_are_black = np.less(avg_inten, blackness_threshold)
    sequence_lengths, scaled_values = extract_preliminary_sequence_lengths(pixels_are_black)
    sequence_lengths, scaled_values = collapse_array(sequence_lengths, scaled_values)
    masks = extract_bookmarks(sequence_lengths, scaled_values)
    return masks



# Some images have 2 logos in one, stacked vertically
# Some are bordered in black space
# This returns an array {1: image1, ...} of just the split and cropped images
def return_split_and_cropped_images(normalized_image):
    vertical_blocks = extract_vertical_clipping_masks(normalized_image, 0)
    extracted_images = {}
    count = 0
    for boundaries in vertical_blocks:
        count += 1
        vertical_crop = normalized_image[boundaries[0]:boundaries[1]]
        horizontal_crop = remove_bounding_box(vertical_crop)
        extracted_images[count] = horizontal_crop
    return extracted_images


def mark_images_for_splitting(normalized_image):
    vertical_blocks = extract_vertical_clipping_masks(normalized_image)
    return vertical_blocks
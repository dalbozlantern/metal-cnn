import os
import pandas as pd
import numpy as np
import PIL.Image

from scipy.ndimage.filters import gaussian_filter
from scipy import misc

from skimage.filters import threshold_otsu
from skimage.filters import rank
from skimage.morphology import disk

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import math

import webbrowser


# Load project variables
from configparser import ConfigParser
config = ConfigParser()
config.read('config.ini')
img_root = config.get('main', 'img_root')

#===========================================================================
# Low-level image utilities

def show_image(image_np_array):
    PIL.Image.fromarray(image_np_array).show()


def load_image(img_file_name):
    return np.array(misc.imread(img_file_name))

def save_image(image_array, dir, file_name):
    misc.toimage(image_array, cmin=0.0, cmax=255).save(dir + '/' + file_name)


#===========================================================================
# Image transformations

# Returns a greyscale image maxed on the maximum intensity across channels
def max_rgb2grey(image):
    if len(image.shape) != 2:
        return np.amax(image[...,:2], 2)
    else:
        return image


# Returns the "standard" greyscale image, averaging across channels
def crt_rgb2grey(image):
    if len(image.shape) != 2:
        return np.dot(image[...,:3], [0.299, 0.587, 0.114])
    else:
        return image


# "Expands" a 2D greyscape back into 3 dimensions via repetition (useful for matplotlib plotting on graphs)
def expand_greyscale(greyscale_2D):
    greyscale_3D = np.expand_dims(greyscale_2D, 2)
    greyscale_3D = np.repeat(greyscale_3D, 3, 2)
    return greyscale_3D


# Takes an image and an intensity threshold and outputs a hard black-and-white stencil
def threshold_split(greyscale_image, threshold):
    black_and_white = np.empty(greyscale_image.shape[:2])
    for x in range(0, greyscale_image.shape[0]):
        for y in range(0, greyscale_image.shape[1]):
            if greyscale_image[x, y] > threshold:
                black_and_white[x, y] = 255
            else:
                black_and_white[x, y] = 0
    return black_and_white


# Similar to threshold_split(), but auto-calculates the optimal threshold
def threshold_image(greyscale_image):
    return threshold_split(greyscale_image, threshold_otsu(greyscale_image))


# Returns the x-axis value of the Otsu threshold of an image (in % of pixels above/below the threshold)
def otsu_percentile(greyscale_image):
    threshold = threshold_otsu(greyscale_image)
    intensity_curve = create_intensity_curve(greyscale_image)
    intensity_difference_from_threshold = [abs(i - threshold) for i in intensity_curve]
    minimum_differences = intensity_difference_from_threshold == min(intensity_difference_from_threshold)
    axis = list(range(len(intensity_curve)))
    axis = [i / (len(intensity_curve) - 1) for i in axis]
    average_minimum = np.dot(minimum_differences, axis) / sum(minimum_differences)
    return average_minimum


# Smooth image
def smooth_image(greyscale_image, blur=1):
    reformatted_greyscale = np.array( [i/255 for i in greyscale_image] )
    return rank.mean_bilateral(reformatted_greyscale, disk(blur), s0=500, s1=500)


# Performs a hard clip and rescaling for black and white
def normalize_image(greyscale_image, cutoff=10):
    bottom_treshold = np.percentile(greyscale_image, cutoff)
    top_threshold = np.percentile(greyscale_image, 100 - cutoff)
    if bottom_treshold == top_threshold:
        return greyscale_image
    capped = np.minimum(greyscale_image, top_threshold)
    capped = np.maximum(capped, bottom_treshold)
    norm = np.add(capped, -bottom_treshold)
    norm = np.multiply(norm, 255 / (top_threshold - bottom_treshold))
    return norm


# Returns an image of an identical size, but with only the border showing (all the interior is black)
def mask_to_border(greyscale_image, border=5):
    height = greyscale_image.shape[0]
    width = greyscale_image.shape[1]
    # Create the border mask
    bar_horiz = np.ones([border, width])
    bar_vert = np.ones([height - 2*border, border])
    center = np.zeros([height - 2*border, width - 2*border])
    middle = np.concatenate((bar_vert, center, bar_vert), 1)
    mask = np.concatenate((bar_horiz, middle, bar_horiz), 0)
    # Extract the border
    border = np.multiply(mask, greyscale_image)
    return border


def remove_bounding_box(normalized_image):
    mask = np.abs(normalized_image - np.array([0])) <= 2.5

    # Find the bounding box of those pixels
    coords = np.array(np.nonzero(~mask))
    top_left = np.min(coords, axis=1)
    bottom_right = np.max(coords, axis=1)

    out = normalized_image[top_left[0]:bottom_right[0],
          top_left[1]:bottom_right[1]]

    return out


#===========================================================================
# Intensity curves


def format_as_percent(x, pos=0):
    return '{0:.0f}%'.format(100*x)


# Returns an array showing the difference between consecutive entries in an input array
def difference_vector(input_array):
    return [input_array[i] - input_array[i - 1] for i in range(1, len(input_array))]


# Takes a greyscale image and turns it into a sorted, 1D pixel intensity curve
def create_intensity_curve(greyscale_image):
    greyscale_img = greyscale_image.copy()
    height = len(greyscale_img)
    width = len(greyscale_img[0])
    greyscale_img.resize((1, height * width))  # 1D: (h*w)
    greyscale_img = greyscale_img[0]  # extract resized array from redundant bracketing
    greyscale_img.sort()  # dark to bright, 0 to 255
    return greyscale_img


# Plots a *FORMATTED* intensity curvey, with the x-axis being "% of sample"
# If an Otsu threshold is also passed, it shows that line
def plot_intensity_curve(formatted_intensity_curve, x_threshold=-1, y_threshold=-1):
    plotting_axis = list(range(len(formatted_intensity_curve)))
    plotting_axis = [i / (len(formatted_intensity_curve) - 1) for i in plotting_axis]

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)
    ax2 = fig1.add_subplot(1, 1, 1)
    ax3 = fig1.add_subplot(1, 1, 1)

    ax1.plot(plotting_axis, formatted_intensity_curve, linewidth=2)

    ax1.set_xlim(0, 1)
    ax1.set_xlabel('% of pixels in image (cumulative)')
    ax1.xaxis.set_major_formatter(FuncFormatter(format_as_percent))

    ax1.set_ylim(0, 255)
    ax1.set_ylabel('Pixel intensity')

    if x_threshold != -1:
        ax2.plot([x_threshold, x_threshold], [0, 255], 'red', linewidth=1)

    if y_threshold != -1:
        ax3.plot([0, 100], [y_threshold, y_threshold], 'red', linewidth=1)

    fig1.show()


# Creates a gaussian smoothed approximation of the slope for an intensity curve
def create_density_curve(formatted_intensity_curve, sigma=3000):
    intensity_difference = difference_vector(formatted_intensity_curve)
    density = gaussian_filter(intensity_difference, sigma)
    return density


# Plots a density curve
def plot_density_curve(formatted_density_curve, x_threshold=-1):
    plotting_axis = list(range(len(formatted_density_curve)))
    plotting_axis = [i / (len(formatted_density_curve) - 1) for i in plotting_axis]

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)
    ax2 = fig1.add_subplot(1, 1, 1)

    ax1.plot(plotting_axis, formatted_density_curve, linewidth=2)

    ax1.set_xlim(0, 1)
    ax1.set_xlabel('% of pixels in image (cumulative)')
    ax1.xaxis.set_major_formatter(FuncFormatter(format_as_percent))

    ax1.set_ylim(0,)
    ax1.set_ylabel('Slope of intensity')

    if x_threshold != -1:
        ax2.plot([x_threshold, x_threshold], [0, max(formatted_density_curve)], 'red', linewidth=1)

    fig1.show()


# Takes an input greyscale image and plots intensity/slope vs. the thresholds
def create_and_plot_curves(greyscale_image):
    print('Calculating thresholds...')
    x_threshold = otsu_percentile(greyscale_image)
    y_threshold = threshold_otsu(greyscale_image)
    print('Computing density curve...')
    intensity_curve = create_intensity_curve(greyscale_image)
    density_curve = create_density_curve(intensity_curve)
    plot_intensity_curve(intensity_curve, x_threshold, y_threshold)
    plot_density_curve(density_curve, x_threshold)


# Plots an arbitrary array as "% of sample"
def plot_arbitrary_array(array, ylim=[-1, -1]):
    plotting_axis = list(range(len(array)))
    plotting_axis = [i / (len(array) - 1) for i in plotting_axis]

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)

    ax1.plot(plotting_axis, array, linewidth=2)

    ax1.set_xlim(0, 1)
    ax1.set_xlabel('% of sample')
    ax1.xaxis.set_major_formatter(FuncFormatter(format_as_percent))

    if ylim[0] != -1:
        ax1.set_ylim(ylim[0], ylim[1])

    fig1.show()


#===========================================================================
# Output analysis

# Shows the outputs of the two greyscaling methods: max and crt
def compare_greys(img_file_name):
    image = load_image(img_file_name)
    show_image(max_rgb2grey(image))
    show_image(crt_rgb2grey(image))


# Shows the auto-thresholding output based on the two greyscaling methods: max and crt
def compare_otsu(img_file_name):
    image = load_image(img_file_name)

    max_greyscale = max_rgb2grey(image)
    crt_greyscale = crt_rgb2grey(image)

    max_stencil = threshold_image(max_greyscale)
    crt_stencil = threshold_image(crt_greyscale)

    show_image(max_stencil)
    show_image(crt_stencil)
    

# Calculates the mean squared "greyness" of an image
# ("greyness" being the furthest distance from either white or black)
def find_blur_score(normalized_image):
    scaled_norm = np.multiply(normalized_image, 1/255)
    diff_from_center = np.abs(np.add(scaled_norm, -.5))
    diff_from_extremes = np.add(-diff_from_center, .5)
    diff_sq = np.power(diff_from_extremes, 2)
    return np.mean(diff_sq)


# Determines a score representing the proportion of the image's 1-pixel border is "absolute white"
def find_border_score(normalized_image):
    border = mask_to_border(normalized_image, 1)
    height = normalized_image.shape[0]
    width = normalized_image.shape[1]
    perim = 2 * (height + width) - 4
    return np.sum(np.abs(border - 255) < 10) / perim


# Returns how much of a normalized image falls into the black/white bands, and how much is in-between
def bucketize_image(normalized_image):
    px = normalized_image.shape[0] * normalized_image.shape[1]
    black = np.sum(np.abs(normalized_image) < 1) / px  # Weird abs term to cover for floating point errors
    white = np.sum(np.abs(normalized_image - 255) < 1) / px  # Weird abs term to cover for floating point errors
    grey = 1 - black - white
    return black, white, grey


# Saves and opens a test page iterating across a list of image files with some parameter (e.g., a score) attached
def save_to_html(output_file, image_list, param_list, number_of_cuts):
    # Initialize
    assert len(image_list) == len(param_list)
    step_size = math.floor(len(image_list) / number_of_cuts)

    # Build the html file
    with open(output_file, 'w') as file:
        file.write('<html>\n<head>\n')
        for i in range(0, number_of_cuts * step_size, step_size):
            link = image_list[i]
            link = link.replace('?', '%3F')
            link = link.replace('#', '%23')
            link = link.replace(' ', '%20')
            file.write('<img height="100" src="' + link + '"><br>\n')
            file.write(str(param_list[i]) + '<br><br>\n')
        file.write('</body>\n</html>')

    # Open the file
    if output_file[0] == '/':
        url = output_file
    else:
        current_working_directory = os.getcwd()
        url = current_working_directory + '/' + output_file
    webbrowser.open(url, new=2)


#===========================================================================
# Scratchwork


# Loading useful samples

rootpath = '/mnt/2Teraz/DL-datasets/metal-cnn-images/test_files/'

def load_examples():
    links = {1: rootpath + '27213_logo.gif',  # irredemable
             2: rootpath + '3540409149_logo.png',  # ideal
             3: rootpath + '3540305397_logo.jpg',  # good but whispy
             4: rootpath + '3540285103_logo.jpg',  # good but blurry
             }
    images = {}
    for i in links:
        images[i] = load_image(links[i])
    greys = {}
    for i in links:
        greys[i] = max_rgb2grey(images[i])
    norms = {}
    for i in links:
        norms[i] = normalize_image(greys[i])
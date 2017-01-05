import numpy as np
import PIL.Image

from scipy.ndimage.filters import gaussian_filter
from scipy import misc

from skimage.filters import threshold_otsu
from skimage.filters import rank
from skimage.morphology import disk

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import os
import pandas as pd

from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import math


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


#===========================================================================
# Intensity curves


def format_as_percent(x, pos=0):
    return '{0:.0f}%'.format(100*x)


# Returns an array showing the difference between consecutive entries in an input array
def difference_vector(input_array):
    return [input_array[i] - input_array[i - 1] for i in range(1, len(input_array))]


# Takes a greyscale image and turns it into a sorted, 1D pixel intensity curve
def create_intensity_curve(input_greyscale_img):
    greyscale_img = input_greyscale_img.copy()
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
def create_and_plot_curves(input_greyscale_img):
    print('Calculating thresholds...')
    x_threshold = otsu_percentile(input_greyscale_img)
    y_threshold = threshold_otsu(input_greyscale_img)
    print('Computing density curve...')
    intensity_curve = create_intensity_curve(input_greyscale_img)
    density_curve = create_density_curve(intensity_curve)
    plot_intensity_curve(intensity_curve, x_threshold, y_threshold)
    plot_density_curve(density_curve, x_threshold)



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
# Image transformations

# Returns a greyscale image maxed on the maximum intensity across channels
def max_rgb2grey(image):
    return np.amax(image[...,:2], 2)


# Returns the "standard" greyscale image, averaging across channels
def crt_rgb2grey(image):
    return np.dot(image[...,:3], [0.299, 0.587, 0.114])


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
    intensity_curve = create_intensity_curve(grey)
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

#===========================================================================
# Scratchwork


# Loading useful samples

rootpath = '/mnt/2Teraz/DL-datasets/metal-cnn-images/test_files/'

img1 = load_image(rootpath + '27213_logo.gif')  # irredemable
grey1 = max_rgb2grey(img1)
img2 = load_image(rootpath + '3540409149_logo.png')  # ideal
grey2 = max_rgb2grey(img2)
img3 = load_image(rootpath + '3540305397_logo.jpg')  # good but whispy
grey3 = max_rgb2grey(img3)
img4 = load_image(rootpath + '3540285103_logo.jpg')  # good but blurry
grey4 = max_rgb2grey(img4)

grey_list = [grey1, grey2, grey3, grey4]


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


def blur_score(normed_image):
    if np.maximum(normed_image) > 1:
        norm = np.multiply(normed_image, 1/255)
    diff_from_center = np.abs(np.add(normed_image, -.5))
    diff_from_extremes = np.add(-diff_from_center, .5)
    diff_sq = np.power(diff_from_extremes, 2)
    return np.mean(diff_sq)


def find_avg_border_color(greyscale_image, border=5):
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
    return np.sum(border) / np.sum(mask)


def test_scores():
    scores = []
    file_names = []
    count = 0
    for root, dirs, files in os.walk(rootpath):
        for name in files:
            count += 1
            if count % 10 == 0:
                print(count)
            if count >= 100:
                break
            try:
                img = load_image(os.path.join(root, name))
                grey = max_rgb2grey(img)
                norm = normalize_image(grey)
                score = blur_score(norm)
                scores += [score]
                file_names += [os.path.join(root, name)]
            except:
                pass
    doo_df2 = pd.DataFrame({'score': scores, 'file': file_names})
    doo_df2 = doo_df2.sort(columns='score')
    doo_df2.index = range(1, len(doo_df2) + 1)
    plot_arbitrary_array(doo_df2['score'])


def sample(i):
    print(doo_df2['score'][i])
    show_image(load_image(doo_df2['file'][i]))
    show_image(max_rgb2grey(load_image(doo_df2['file'][i])))




def expand_greyscale(greyscale_2D):
    greyscale_3D = np.expand_dims(greyscale_2D, 2)
    greyscale_3D = np.repeat(greyscale_3D, 3, 2)
    return greyscale_3D

def plot_image(image, scale, x, y):
    if len(image.shape) == 2:
        image = expand_greyscale(image)
    imagebox = OffsetImage(image, zoom=scale)
    ab = AnnotationBbox(imagebox, [x, y],
                        xybox=(30., -30.),
                        xycoords='data',
                        boxcoords="offset points")
    return ab


def plot_examples(n=100, k=11):
    fig = plt.gcf()
    fig.clf()
    ax = plt.subplot(111)

    j = math.floor((n-1)/(k-1))
    for i in range(1, k):
        img = load_image(doo_df2['file'][1 + i*j])
        ax.add_artist(plot_image(img, .25, (i)/(k), .5))

    ax.grid(False)
    plt.draw()
    plt.show()


bkgs = []
files = []
count = -1
for root, dirs, files in os.walk(img_root + '/0000'):
    for name in files:
        count += 1
        if count % 100 == 0:
            print(str(count))
        try:
            img = load_image(os.path.join(root, name))
            grey = max_rgb2grey(img)
            norm = normalize_image(grey)
            bkg = find_avg_border_color(norm)
            bkgs += [bkg]
            files += [os.path.join(root, name)]
        except:
            pass
border_df = pd.DataFrame({'file': files, 'bkg': bkgs})
plot_arbitrary_array(bkgs)
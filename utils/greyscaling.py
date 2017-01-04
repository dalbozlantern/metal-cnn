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



sample_files = ['126409_logo.jpg',
             '27213_logo.gif',
             '3540262037_logo.jpg',
             '3540270735_logo.jpg',
             '3540285103_logo.jpg',
             '3540298623_logo.jpg',
             '3540305397_logo.jpg',
             '3540345632_logo.jpg',
             '3540354438_logo.jpg',
             '3540366370_logo.jpg',
             '3540367708_logo.jpg',
             '3540383763_logo.png',
             '3540388380_logo.jpg',
             '3540389754_logo.gif',
             '3540397209_logo.jpg',
             '3540409149_logo.png',
             '3540418516_logo.jpg',
             '46351_logo (1).gif',
             '46351_logo.gif',
             '58202_logo.JPG',
             '6164_logo.jpg',
             '69184_logo.jpg',
             '79251_logo.jpg',
             '87179_logo.jpg',
             '87712_logo.GIF',
             '94913_logo.jpg',
             '95296_logo.jpg',
             'iron.png']

rootpath = '/mnt/2Teraz/DL-datasets/metal-cnn-images/test_files/'

def iterate_over_sample(sample_files, rootpath):
    for file in sample_files:
        file_name = rootpath + file
        image = load_image(file_name)
        greyscale_image = max_rgb2grey(image)
        print(file_name)
        show_image(greyscale_image)
        score = blur_score(greyscale_image)
        print(score)
        input("Press Enter to continue...")



img1 = load_image(rootpath + '27213_logo.gif')  # irredemable
grey1 = max_rgb2grey(img1)
img2 = load_image(rootpath + '3540409149_logo.png')  # ideal
grey2 = max_rgb2grey(img2)
img3 = load_image(rootpath + '3540305397_logo.jpg')  # good but whispy
grey3 = max_rgb2grey(img3)
img4 = load_image(rootpath + '3540285103_logo.jpg')  # good but blurry
grey4 = max_rgb2grey(img4)


grey_list = [grey1, grey2, grey3, grey4]

def img_diag(grey):
    return ((len(grey)**2+len(grey[0])**2)**.5)

def test_me(gamma=10):
    for grey in grey_list:
        bottom = np.percentile(grey, gamma)
        top = np.percentile(grey, 100-gamma)
        clip = np.empty(grey.shape[:2])
        for x in range(0, grey.shape[0]):
            for y in range(0, grey.shape[1]):
                norm = min(max(grey[x, y], bottom), top)
                norm = (norm - bottom) / (top - bottom)
                # norm = .5 - abs(.5 - norm)
                clip[x, y] = norm
        diff = np.add(-clip, .5)
        diff = np.abs(diff)
        diff = np.add(-diff, .5)
        diff = np.multiply(diff, 2)
        diff_sq = np.power(diff, 2)
        print(np.sum(diff_sq))
        slice = clip[len(clip) / 3]
        plot_arbitrary_array(slice)

test_me()

def white_score(greyscale_image):
    threshold = threshold_otsu(greyscale_image)
    intensity_curve = create_intensity_curve(greyscale_image)
    white_curve_portion = [i for i in intensity_curve if i >= threshold]
    white_differences = difference_vector(white_curve_portion)
    avg_slope = sum(white_differences) / len(white_differences)
    return avg_slope





#
def blur_score(greyscale_image, cutoff=2):
    bottom_treshold = np.percentile(greyscale_image, cutoff)
    top_threshold = np.percentile(greyscale_image, 100 - cutoff)
    if bottom_treshold == top_threshold:
        return 1
    clip = np.empty(greyscale_image.shape[:2])
    for x in range(0, greyscale_image.shape[0]):
        for y in range(0, greyscale_image.shape[1]):
            norm = min(max(greyscale_image[x, y], bottom_treshold), top_threshold)
            norm = (norm - bottom_treshold) / (top_threshold - bottom_treshold)
            norm = .5 - abs(.5 - norm)
            clip[x, y] = norm
    return np.mean(clip)


def blur_score2(greyscale_image, cutoff=10):
    bottom_treshold = np.percentile(greyscale_image, cutoff)
    top_threshold = np.percentile(greyscale_image, 100 - cutoff)
    if bottom_treshold == top_threshold:
        return 1
    capped = np.minimum(greyscale_image, top_threshold)
    capped = np.maximum(capped, bottom_treshold)
    norm = np.add(capped, -bottom_treshold)
    norm = np.multiply(norm, 1/(top_threshold - bottom_treshold))
    diff_from_center = np.abs(np.add(norm, -.5))
    diff_from_extremes = np.add(-diff_from_center, .5)
    diff_sq = np.power(diff_from_extremes, 2)
    return np.mean(diff_sq)


scores = []
file_names = []
count = 0
for root, dirs, files in os.walk('/mnt/2Teraz/DL-datasets/metal-cnn-images/01-raw'):
    for name in files:
        count += 1
        if count % 10 == 0:
            print(count)
        if count >= 100:
            break
        try:
            img = load_image(os.path.join(root, name))
            grey = max_rgb2grey(img)
            score = blur_score2(grey)
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



from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import math


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



fig = plt.gcf()
fig.clf()
ax = plt.subplot(111)

n = 100
k = 11
j = math.floor((n-1)/(k-1))
for i in range(1, k):
    img = load_image(doo_df2['file'][1 + i*j])
    ax.add_artist(plot_image(img, .25, (i)/(k), .5))

ax.grid(False)
plt.draw()
plt.show()
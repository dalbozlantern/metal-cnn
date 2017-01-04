import numpy as np
from PIL import Image

from scipy.ndimage.filters import gaussian_filter
from scipy import misc

from skimage.filters import threshold_otsu
from skimage.filters import rank
from skimage.morphology import disk

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


#===========================================================================
# Low-level image utilities

def show_image(image_np_array):
    Image.fromarray(image_np_array).show()


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


#===========================================================================
# Image transformations

# Returns a greyscale image maxed on the maximum intensity across channels
def max_rgb2grey(image):
    max_grey = np.empty(image.shape[:2])
    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            max_grey[x, y] = image[x, y][:3].max()
    return max_grey


# Returns the "standard" greyscale image, averaging across channels
def crt_rgb2grey(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


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
             'iron.png',
             'plot_canny.py']

rootpath = '/mnt/2Teraz/DL-datasets/metal-cnn-images/test_files/'

def iterate_over_sample(sample_files, rootpath):
    for file in sample_files:
        file_name = rootpath + file
        image = load_image(file_name)
        greyscale_image = max_rgb2grey(image)
        optimized_image = threshold_image(greyscale_image)
        smooth_out = smooth_image(optimized_image)
        print(file_name)
        # create_and_plot_curves(greyscale_image)
        show_image(image)
        # show_image(greyscale_image)
        # show_image(optimized_image)
        # print(white_score(greyscale_image))
        # show_image(smooth_out)
        # input("Press Enter to continue...")



img = load_image(rootpath + '3540305397_logo.jpg')
grey = max_rgb2grey(img)
thr = threshold_image(grey)
show_image(thr)

from skimage.restoration import denoise_nl_means
show_image(denoise_nl_means(thr, 7, 11, 0.1, False))



# return the otsu threshold
# return the curve
# return the curve above the otsu threshold
# make a difference vector
# average the difference vector
greyscale_image = []

def white_score(greyscale_image):
    threshold = threshold_otsu(greyscale_image)
    intensity_curve = create_intensity_curve(greyscale_image)
    white_curve_portion = [i for i in intensity_curve if i >= threshold]
    white_differences = difference_vector(white_curve_portion)
    avg_slope = sum(white_differences) / len(white_differences)
    return avg_slope

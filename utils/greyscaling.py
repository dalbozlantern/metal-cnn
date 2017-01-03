import numpy as np
from PIL import Image
from scipy import misc
from skimage.filters import threshold_otsu


def dan_rgb2grey(image):
    dan_grey = np.empty(image.shape[:2])
    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            dan_grey[x, y] = image[x, y][:3].max()
    return dan_grey

def threshold_split(image, threshold):
    black_and_white = np.empty(image.shape[:2])
    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            if image[x, y] > threshold:
                black_and_white[x, y] = 255
            else:
                black_and_white[x, y] = 0
    return black_and_white


def crt_rgb2grey(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def show_image(image):
    Image.fromarray(image).show()


def compare_greys(img_location):
    image = np.array(misc.imread(img_location))
    show_image(dan_rgb2grey(image))
    show_image(crt_rgb2grey(image))


def threshold_image(image):
    return threshold_split(image, threshold_otsu(image))


def compare_otsu(img_location):
    image = np.array(misc.imread(img_location))
    show_image(threshold_image(dan_rgb2grey(image)))
    show_image(threshold_image(crt_rgb2grey(image)))
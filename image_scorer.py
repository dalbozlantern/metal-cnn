import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import gaussian_kde
import os
from metal_utils.widgets import progress_bar


def plot_array(data):
    plotting_axis = list(range(len(data)))
    plotting_axis = [100 * i / (len(data) - 1) for i in plotting_axis]
    plt.plot(plotting_axis, data, linewidth=2)
    plt.xlim(0, 100)
    plt.ylim(0, )
    plt.show()
    return

def density_plot(data):
    density = gaussian_kde(data)
    xs = np.linspace(0, 1, 200)
    density.covariance_factor = lambda: .25
    density._compute_covariance()
    plt.plot(xs, density(xs))
    plt.show()


def extract_score(file_name, steps=False):
    img = mpimg.imread(file_name)  # (h, w, c)
    img = img[:3]  # ditch the alpha channel if gif
    if steps:
        Image.fromarray(img).show()

    img = np.amax(img, 2)  # (h, w)
    if steps:
        Image.fromarray(img).show()

    height = len(img)
    width = len(img[0])
    img.resize((1, height * width))  # (h*w)
    img = img[0]  # extract resized array from redundant bracketing
    img.sort()  # dark to bright, 0 to 255
    if steps:
        plot_array(img)

    deriv = [float(img[i] - img[i - 1]) for i in range(1, height * width)]
    if steps:
        density_plot(deriv)

    avg_deriv = sum(deriv) / len(deriv)
    return avg_deriv


img_list = []
scores = []
count = 0
for dir, subdir, files in os.walk('01-raw/'):
    num_files = len(files)
    for file in files:
        try:
            count += 1
            if count % 10 == 0:
                progress_bar('Image 3', count, num_files)
            filename = '01-raw/' + file
            print(filename)
            score = extract_score(filename)
            scores += [score]
            img_list += ['01-raw/' + file]

        except:
            pass
scores_sorted = list(scores)
scores_sorted.sort()
plot_array(scores_sorted)

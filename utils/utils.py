# =========================================================================
# Description: This file contains several misc. utilities
# =========================================================================

import smtplib
import matplotlib.pyplot as plt

# =========================================================================



# This function displays a progress bar that looks like this:
# 84%  ||#####|#####|#####|##   ||  (Document number 218339 / 258838)...
def progress_bar(label, currentN, endN, startN=0):
    progress = (currentN - startN) / (endN - startN)
    progress_format = '%02d' % (progress * 100) + '%'
    num_hashes = int(round(progress*20, 0))
    hash1 = '||' + '#' * max(0, min(num_hashes, 5)) + ' ' * (5 - max(0, min(num_hashes, 5)))
    hash2 = '|' + '#' * max(0, min(num_hashes - 5, 5)) + ' ' * (5 - max(0, min(num_hashes - 5, 5)))
    hash3 = '|' + '#' * max(0, min(num_hashes - 10, 5)) + ' ' * (5 - max(0, min(num_hashes - 10, 5)))
    hash4 = '|' + '#' * max(0, min(num_hashes - 15, 5)) + ' ' * (5 - max(0, min(num_hashes - 15, 5)))
    progressbar = hash1 + hash2 + hash3 + hash4 + '||'
    print('    {0}  {1}  ({2} {3} / {4})...'.format(progress_format, progressbar, label, currentN, endN), end="\r")


def email_alert(msg):
    email_info = open('text_cleaning/sec.pw', 'r').read().split('\n')
    to_address = email_info[0]
    pw = email_info[1]
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login('rezrovlab@gmail.com', pw)
    server.sendmail('rezrovlab@gmail.com', to_address, msg)
    server.quit()
    return


def format_as_percent(x, pos=0):
    return '{0:.0f}%'.format(100*x)


# Plots an arbitrary array as "% of sample"
def plot_arbitrary_array(array, y_label=None, y_log=False, ylim=[-1, -1]):
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
    if y_label:
        ax1.set_ylabel(y_label)
    if y_log:
        ax1.set_yscale('log')

    fig1.show()
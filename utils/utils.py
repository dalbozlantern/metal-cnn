# =========================================================================
# Description: This file contains several misc. utilities
# =========================================================================

import smtplib

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
# -*- coding: utf-8 -*-

"""
Generates templates with dots for each domino
"""

import numpy as np
import matplotlib as mpl
from matplotlib.image import imsave
import cv2 as cv
import os

folder = "domino_templ"
base_images = []
final_images = []
for i in range(1, 7):
    img = cv.imread(os.path.join(folder, "base/%i.png" %i))
    if img is not None:
        img = img[:, :, 0]
        base_images.append(img)
        final_images.append(img)
        imsave(folder + '/0%i.png' %i, 255 - img, cmap='binary')

for i in range(6):
    for j in range(i + 1):
        upper_half = base_images[i][:, :]
        lower_half = base_images[j][::-1, :]
        img = np.asarray(np.minimum(upper_half, lower_half), dtype='uint8')
        imsave(folder + '/%i%i.png' % (j+1, i+1), 255 - img, cmap='binary')

# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 15:53:46 2019

@author: anana
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2 as cv
import os



# load images
folder  = "images"
images = []
for filename in os.listdir(folder):
    img = cv.imread(os.path.join(folder,filename))
    if img is not None:
        images.append(img)
bwImg = []
edgeImg = []
colorImg = []
contoursAll = []
areas = []

# figEdge, axEdge = plt.subplots(5, 6, sharex=True, sharey=True, dpi=300)
# figBw, axBw = plt.subplots(5, 6, sharex=True, sharey=True, dpi=300)
figIm, axIm = plt.subplots(1, 3, dpi = 300)

# preprosessing
contrast = 1.0
for i in range(len(images)):
    img = images[i]
    img = cv.resize(img, (200, 266))
    # change constrast
    img = np.minimum(255*np.ones((266, 200, 3)), img*contrast)
    # make colorfull pixels darker
    pixelStd = np.dstack((2*np.std(img, axis = 2), )*3)
    img = np.maximum(255*np.zeros((266, 200, 3)), img - pixelStd)
    img = np.asarray(img, dtype="uint8" )
    # turn image to grayscale
    bw = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # set background to black
    (thres, bw) = cv.threshold(bw, 70, 255, cv.THRESH_TOZERO)
    # canny edge 
    edge = cv.Canny(bw, 200, 200)
    # bw = cv.adaptiveThreshold(bw, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #                            cv.THRESH_BINARY,101,2)
    (thres, bw) = cv.threshold(bw,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    (im2, contours, hierarchy) = cv.findContours(bw, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    dominoContours = []
    for cont in contours:
        if (cv.contourArea(cont) > 400
            and cv.contourArea(cont) < 1800
            and cv.isContourConvex):
            dominoContours.append(cont)
            areas.append(cv.contourArea(cont))
            rect = cv.minAreaRect(cont)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            cv.drawContours(img,[box],0,(0,0,255),2)
            # check for the color?
    # cv.drawContours(img, dominoContours, -1, (0,200,0), 1)
    bwImg.append(bw)
    edgeImg.append(edge)
    colorImg.append(img)
    contoursAll.append(dominoContours)
    # axBw[i//6, i%6].imshow(bw, cmap='binary')
    # axEdge[i//6, i%6].imshow(edge, cmap='binary')
    

# display one image on a large axis for further examination
displayID = 11
# axIm[0].imshow(edgeImg[displayID], cmap='binary')
axIm[0].imshow(colorImg[displayID])
axIm[0].set_title(r'Starting image')
axIm[1].imshow(bwImg[displayID], cmap='binary')
axIm[1].set_title('Bitmap image')
axIm[2].imshow(edgeImg[displayID], cmap='binary')
axIm[2].set_title('Canny edges')

fig, ax = plt.subplots(1, 1, dpi = 300)
n, bins, patches = plt.hist(areas, 500, density=True)

for ax in axIm:
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])


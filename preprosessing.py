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

figEdge, axEdge = plt.subplots(5, 6, sharex=True, sharey=True, dpi=300)
figBw, axBw = plt.subplots(5, 6, sharex=True, sharey=True, dpi=300)
figIm, axIm = plt.subplots(1, 2, dpi = 300)

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
    images[i] = img
    # turn image to grayscale
    bw = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # set background to black
    (thres, bw) = cv.threshold(bw, 70, 255, cv.THRESH_TOZERO)
    # canny edge 
    edge = cv.Canny(bw, 200, 200)
    # bw = cv.adaptiveThreshold(bw, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #                            cv.THRESH_BINARY,101,2)
    (thres, bw) = cv.threshold(bw,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    # (im2, contours, hierarchy) = cv.findContours(bw, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # cv.drawContours(img, contours, -1, (0,200,0), 1)
    bwImg.append(bw)
    edgeImg.append(edge)
    axBw[i//6, i%6].imshow(bw, cmap='binary')
    axEdge[i//6, i%6].imshow(edge, cmap='binary')
    


# display one image on a large axis for further examination
displayID = 11
# axIm[0].imshow(edgeImg[displayID], cmap='binary')
axIm[0].imshow(bwImg[displayID], cmap='binary')
axIm[1].imshow(images[displayID])
axIm[0].xaxis.set_ticks([])
axIm[0].yaxis.set_ticks([])
axIm[1].xaxis.set_ticks([])
axIm[1].yaxis.set_ticks([])


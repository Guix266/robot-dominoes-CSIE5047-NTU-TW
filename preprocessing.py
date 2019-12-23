# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import label
import cv2 as cv
import os

def fill_holes(img):
    imageCopy = img.copy()
    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv.floodFill(imageCopy, mask, (0,0), 255);
    imageCopy = cv.bitwise_not(imageCopy)
    result = img | imageCopy
    return result

def choose_rand_centroids(blob, k):
    blob_area = np.sum(blob)
    rand_ind = np.random.choice(blob_area, k)
    rand_coor = []
    m = 0
    for i in range(blob.shape[0]):
        for j in range(blob.shape[1]):
            if blob[i, j] == True:
                m += 1
                if m in rand_ind:
                    rand_coor.append((i, j))
    return np.array(rand_coor)


def create_template(area, ratio=1.85):
    """create the basic pattern of a domino"""
    w = int(round(min(np.sqrt(area/ratio), np.sqrt(area*ratio))))
    h = int(round(max(np.sqrt(area/ratio), np.sqrt(area*ratio))))
    diag = int(round(np.sqrt(w**2 + h**2)))
    # make diag odd
    if diag % 2 == 0:
        diag += 1
    origin = np.array([diag // 2 + 1, diag // 2 + 1])
    x_min = origin[0] - h//2
    x_max = origin[0] + h//2 + h%2
    y_min = origin[1] - w//2
    y_max = origin[1] + w//2 + w%2
    templ_simple = np.zeros((diag, diag), dtype='uint8')
    templ_simple[x_min:x_max, y_min:y_max] = 1
    return(templ_simple, origin, diag)


def generate_templates(templ_simple, origin, diag, num=24):
    """Reproduce the templates with rotations"""
    templates = []
    # find the center of the matrix
    for i in range(num):
        phi = 2 * np.pi / num * i
        rot = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        templ = np.zeros((diag, diag), dtype='uint8')
        for x in range(diag):
            for y in range(diag):
                if templ_simple[x, y] == 1:
                    xn, yn = rot.dot(np.array([x, y]) - origin) + origin
                    xn = int(round(xn, 0))
                    yn = int(round(yn, 0))
                    try:
                        templ[xn, yn] = 1
                    except IndexError:
                        continue
        # add an empty line at every border to fill the holes
        temp = np.zeros((diag + 2, diag + 2), dtype='uint8')
        temp[1:-1, 1:-1] = templ
        temp = fill_holes(temp)
        templates.append(temp)
    return templates



def segment(img, N, threshold=70000):
    """
    Parameters
    ----------
    img : binary array
        A binary image of black siluets of dominoes on white background.
    N : int
        Expected number of dominoes
    threshold : float
        thresheld for template matching

    Returns
    -------
    coordinates : list
        A list of coordinates of the centroids and the rotation angle in radian with respect to the y-axis
    """
    # approx height to width ratio of a domino
    ratio = 1.85
    labeled_img = np.zeros(bw.shape)
    img = (img == np.max(img))
    total_area = np.sum(img)
    tile_area = total_area/N
    # label all shapes in the image
    (img, m) = label(img)
    # generate a list of rotated dominoes for template matching
    (templ_simple, origin, diag) = create_template(tile_area, ratio=1.85)
    plt.imshow(templ_simple)
    templates = generate_templates(templ_simple, origin, diag, num=40)
    # approx width and height of a domino
    w = int(round(min(np.sqrt(tile_area / ratio), np.sqrt(tile_area * ratio))))
    h = int(round(max(np.sqrt(tile_area / ratio), np.sqrt(tile_area * ratio))))
    coordinates = []
    # if the number of blobs matches the expected number of dominoes, we're done
    if m == N:
        for i in range(m):
            blob = np.asarray((img == i), dtype='uint8')
            M = cv.moments(blob)
            xb = int(round(M['m01'] / M['m00']))
            yb = int(round(M['m10'] / M['m00']))
            phi = 0.5 * np.arctan2(2 * M['mu11'], (M['mu20'] - M['mu02']))
            coordinates.append([xb, yb, phi])
        return coordinates
    # but it doesn't
    for i in range(1, m+1):
        # separate a blob
        blob = np.asarray((img == i), dtype='uint8')
        blob_area = np.sum(blob)/np.max(blob)
        if np.abs(blob_area - tile_area) < tile_area*0.5:
            M = cv.moments(blob)
            xb = int(round(M['m01'] / M['m00']))
            yb = int(round(M['m10'] / M['m00']))
            phi = 0.5*np.arctan2(2*M['mu11'], (M['mu20'] - M['mu02']))
            coordinates.append([xb, yb, phi])
            continue
        else:
            # First, try to match a set of rotated domino templates to the blob. Find the best match among all
            # coordinates and all templates. Since the templates have a clean background, this should find the
            # outermost domino. Then, save the expected centroid and angle of the domino and subtract the template
            # from the blob. Repeat till all dominoes are found.
            approx_coordinates = []
            while blob_area > tile_area*0.8:
                best_match = np.zeros((blob.shape[0] - templates[0].shape[0] + 1,
                                       blob.shape[1] - templates[0].shape[1] + 1,
                                       len(templates)))
                for i in range(len(templates)):
                    templ = templates[i]
                    matching_result = cv.matchTemplate(np.asarray(blob, dtype='float32'),
                                                       np.asarray(templ, dtype='float32'),
                                                       method=cv.TM_CCOEFF)
                    best_match[:, :, i] = np.where(matching_result >= threshold, matching_result, 0)
                # find the best result
                (xopt, yopt, k) = np.unravel_index(np.argmax(best_match, axis=None), best_match.shape)
                if xopt == 0 and yopt == 0:
                    print("Can't find any other tiles!")
                    break
                # angle of rotation of the template
                phi = 2*np.pi/len(templates)*k
                # save approximate coordinates of the centroid
                xb = xopt + templates[0].shape[0] // 2
                yb = yopt + templates[0].shape[1] // 2
                approx_coordinates.append([xb, yb, phi])
                # subtract the best matching template from the blob
                best_templ = templates[k]
                mask = np.zeros(img.shape, dtype=best_templ.dtype)
                mask[xopt:xopt+best_templ.shape[0],
                     yopt:yopt+best_templ.shape[1]] = best_templ
                blob = np.asarray(blob > mask, dtype='uint8')
                blob_area = np.sum(blob) / np.max(blob)
            # TODO: do some processing on approx_coordinates
            coordinates += approx_coordinates
    return coordinates
            
    
# load images
folder = "images"
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
figIm, axIm = plt.subplots(1, 3, dpi=300)

# preprocessing
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
    bw = cv.GaussianBlur(bw, (0,0), 2)
    # set background to black
    (thres, bw) = cv.threshold(bw, 70, 255, cv.THRESH_TOZERO)
    # canny edge 
    edge = cv.Canny(bw, 200, 200)
    # bw = cv.adaptiveThreshold(bw, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #                            cv.THRESH_BINARY,101,2)
    # binary image
    (thres, bw) = cv.threshold(bw,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    # fill the contours
    # issue: fills background enclosed by dominoes
    bw = fill_holes(bw)
    bw_small = cv.resize(bw, dsize=(50, int(266*0.25)))
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
            cv.drawContours(img, [box], 0, (0, 0, 255), 2)
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
N = 12
# axIm[0].imshow(edgeImg[displayID], cmap='binary')
axIm[0].imshow(colorImg[displayID])
axIm[0].set_title(r'Starting image')
axIm[1].imshow(bwImg[displayID], cmap='binary')
axIm[1].set_title('Bitmap image')
axIm[2].imshow(edgeImg[displayID], cmap='binary')
axIm[2].set_title('Canny edges')
coordinates = np.array(segment(bwImg[displayID], N))
axIm[1].plot(coordinates[:, 1], coordinates[:, 0], 'rx')


for ax in axIm:
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])


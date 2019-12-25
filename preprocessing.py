# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import label
import cv2 as cv
import os

def fill_holes(img):
    """
    Fills holes in a binary image
    Parameters
    ----------
    img - binary image

    Returns
    -------
    A copy of the image with filled holes
    """
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


def create_rect_template(area, ratio=1.85):
    """
    Creates an image of a rectangle shape with h:w = ratio to use it as a template for pattern matching.
    The image size is large enough to fit every possible rotation of the rectangle about the center.
    """
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


def generate_rot_templates(templ_simple, origin, start=0, end=2*np.pi, num=24, fill=True):
    """
    Generates a list of rotated templates for pattern matching

    Parameters
    ----------
    templ_simple : uint8 2D-array
        A square grayscale template
    origin : np.array([int, int])
        Coordinates of the rotation point
    start : float
        Starting angle
    end : float
        Final angle
    num : int
        Number of rotated templates

    Returns
    -------
    A list of num rotated templates
    """
    templates = []
    size = templ_simple.shape[0]
    # find the center of the matrix
    for i in range(num):
        phi = start + (end - start) / num * i
        rot = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        templ = np.zeros((size, size), dtype='uint8')
        for x in range(size):
            for y in range(size):
                if templ_simple[x, y] > 0:
                    xn, yn = rot.dot(np.array([x, y]) - origin) + origin
                    xn = int(round(xn, 0))
                    yn = int(round(yn, 0))
                    try:
                        templ[xn, yn] = templ_simple[x, y]
                    except IndexError:
                        continue
        # add an empty line at every border to fill the holes
        temp = np.zeros((size + 2, size + 2), dtype='uint8')
        temp[1:-1, 1:-1] = templ
        if fill:
            temp = fill_holes(temp)
        templates.append(temp)
    return templates

def match_domino(img, coor, tile_area):
    templ_tile_area = 11440 # the area of a 00 domino
    scale_ratio = np.sqrt(tile_area/templ_tile_area)
    phi = coor[2]
    delta_phi = np.pi/30
    #delta_phi = 0
    best_match = 0
    dom_results = {}
    img = img*255/np.max(img)
    # for i in range(7):
    for i in [5, ]:
        # for j in range(i+1):
        for j in [1, ]:
            domino_img = cv.imread("domino_templ/%i%i.png" %(j, i))
            domino_img = cv.cvtColor(domino_img, cv.COLOR_BGR2GRAY)
            domino_img = cv.resize(domino_img, (int(domino_img.shape[0]*scale_ratio),
                                                int(domino_img.shape[1]*scale_ratio)))
            domino_img * 255 / domino_img.max()
            origin = np.array(domino_img.shape, dtype=int)//2
            templ_rot = generate_rot_templates(domino_img,
                                               origin,
                                               phi - delta_phi,
                                               phi + delta_phi,
                                               num=3, fill=False)
            templ_rot_rev = generate_rot_templates(domino_img,
                                                   origin,
                                                   np.pi + (phi - delta_phi),
                                                   np.pi + (phi + delta_phi),
                                                   num=3, fill=False)
            for templ in templ_rot + templ_rot_rev:
                matching_result = cv.matchTemplate(np.asarray(img, dtype='float32'),
                                                   np.asarray(templ, dtype='float32'),
                                                   method=cv.TM_CCORR)

                if (best_match < np.max(matching_result)):
                    best_match = np.max(matching_result)
                    best_matching_result = matching_result
                    best_dom = "%i%i" %(i, j)
                    best_templ = templ
                    (xopt, yopt) = np.unravel_index(np.argmax(best_matching_result, axis=None),
                                                    best_matching_result.shape)
                print("%i%i : %f" % (i, j, np.max(matching_result)))
    return best_dom, (xopt, yopt), best_matching_result, best_templ



def segment(img, N, threshold=70000):
    """
    Attempts to segment a binary image into single rectangular dominoes and determine their angle and coordinates.

    Parameters
    ----------
    img : binary array
        A binary image of black siluets of dominoes on white background.
    N : int
        Expected number of dominoes
    threshold : float
        threshold for template matching

    Returns
    -------
    coordinates : list
        A list of coordinates of the centroids and the rotation angle in radian with respect to the y-axis
    tile_area : int
        An approximate size of a single domino
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
    (templ_simple, origin, diag) = create_rect_template(tile_area, ratio=2)
    plt.imshow(templ_simple)
    templates = generate_rot_templates(templ_simple, origin, num=40)
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
                xb = int(xopt + templates[0].shape[0] // 2)
                yb = int(yopt + templates[0].shape[1] // 2)
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
    return (coordinates, tile_area)
            
    
# load images
# folder = "images"
# images = []
# for filename in os.listdir(folder):
#     img = cv.imread(os.path.join(folder, filename))
#     if img is not None:
#         images.append(img)

bwImg = []
binaryImg = []
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
    # bw = cv.GaussianBlur(bw, (0, 0), 2)
    # set background to black
    (thres, bw) = cv.threshold(bw, 70, 255, cv.THRESH_TOZERO)
    # canny edge 
    edge = cv.Canny(bw, 200, 200)
    # bw = cv.adaptiveThreshold(bw, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #                            cv.THRESH_BINARY,101,2)
    # binary image
    (thres, binary) = cv.threshold(bw, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    # fill the contours
    # issue: fills background enclosed by dominoes
    binary = fill_holes(binary)
    binary_small = cv.resize(binary, dsize=(50, int(266*0.25)))
    (im2, contours, hierarchy) = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
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
    binaryImg.append(binary)
    edgeImg.append(edge)
    colorImg.append(img)
    contoursAll.append(dominoContours)
    # axBw[i//6, i%6].imshow(binary, cmap='binary')
    # axEdge[i//6, i%6].imshow(edge, cmap='binary')


    

# display one image on a large axis for further examination
displayID = 12
N = 12
# axIm[0].imshow(edgeImg[displayID], cmap='binary')
axIm[0].imshow(bwImg[displayID], cmap='binary')
axIm[0].set_title(r'Starting image')
axIm[1].imshow(binaryImg[displayID], cmap='binary')
axIm[1].set_title('Bitmap image')
axIm[2].imshow(edgeImg[displayID], cmap='binary')
axIm[2].set_title('Canny edges')

coordinates, tile_area = segment(binaryImg[displayID], N)
axIm[1].plot(np.array(coordinates)[:, 1], np.array(coordinates)[:, 0], 'rx')

for ax in axIm:
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])

segmentSize = int(np.sqrt(tile_area)*2)

for coor in coordinates:
    segmentVert = [max(coor[0] - segmentSize//2, 0), min(coor[0] + segmentSize//2, 266),
                   max(coor[1] - segmentSize//2, 0), min(coor[1] + segmentSize//2, 200)]
    imSegment = bwImg[displayID][segmentVert[0]:segmentVert[1],
                                 segmentVert[2]:segmentVert[3]]
    imSegment = np.where(imSegment > 120, imSegment, 0)
    # imSegment = imSegment * 255. / np.max(imSegment)
    segmentCoor = [coor[0] - segmentVert[0], coor[1] - segmentVert[2], coor[2]]
    best_dom, (xopt, yopt), result, templ = match_domino(imSegment, segmentCoor, tile_area)

# fig, ax = plt.subplots(1, 3, dpi=300)
ax[0].imshow(imSegment)

templIm = np.zeros(imSegment.shape)
templIm[xopt:xopt+templ.shape[0], yopt:yopt+templ.shape[1]] = templ

ax[1].imshow(templIm)

ax[2].imshow(templIm*imSegment)
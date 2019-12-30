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
    cv.floodFill(imageCopy, mask, (0, 0), 255);
    imageCopy = cv.bitwise_not(imageCopy)
    result = img | imageCopy
    return result

# def choose_rand_centroids(blob, k):
#     blob_area = np.sum(blob)
#     rand_ind = np.random.choice(blob_area, k)
#     rand_coor = []
#     m = 0
#     for i in range(blob.shape[0]):
#         for j in range(blob.shape[1]):
#             if blob[i, j] == True:
#                 m += 1
#                 if m in rand_ind:
#                     rand_coor.append((i, j))
#     return np.array(rand_coor)


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
    templ_simple[x_min:x_max, y_min:y_max] = 255
    return(templ_simple, origin, diag)


def generate_rot_templates(templ_simple, start=0, end=2*np.pi, num=24):
    templates = []
    origin = tuple(np.array(templ_simple.shape[1::-1]) / 2)
    for i in range(num):
        phi = start + (end - start) / num * i
        rot_mat = cv.getRotationMatrix2D(origin, 180*phi/np.pi, 1.0)
        result = cv.warpAffine(templ_simple, rot_mat, templ_simple.shape[1::-1], flags=cv.INTER_LINEAR)
        templates.append(result)
    return templates


def generate_rot_templates_old(templ_simple, origin, start=0, end=2*np.pi, num=24, fill=True):
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
    fill : bool
        Fill holes in the image or not

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
    best_dom = None
    xopt, yopt, best_phi = (0, 0, 0)
    best_matching_result = 0
    best_templ = None

    templ_tile_area = 11440 # the area of a 00 domino
    scale_ratio = np.sqrt(tile_area/templ_tile_area)
    phi = coor[2]
    delta_phi = 0
    best_match = 0
    dom_results = []
    domino_mask = cv.imread("domino_templ/base_mask/mask.png")
    domino_mask = 255 - domino_mask[:, :, 2]
    domino_mask = cv.resize(domino_mask, (int(domino_mask.shape[0] * scale_ratio),
                                          int(domino_mask.shape[1] * scale_ratio)))
    img = img*255/np.max(img)
    origin = np.array(domino_mask.shape, dtype=int) // 2
    templ_mask = generate_rot_templates(domino_mask,
                                        phi - delta_phi,
                                        phi + delta_phi,
                                        num=1)
    templ_mask_rev = generate_rot_templates(domino_mask,
                                            -np.pi + phi - delta_phi,
                                            -np.pi + phi + delta_phi,
                                            num=1)

    for i in range(7):
        for j in range(i+1):
            domino_img = cv.imread("domino_templ/%i%i.png" % (j, i))
            domino_img = domino_img[:, :, 1]
            domino_img = cv.resize(domino_img, (int(domino_img.shape[0]*scale_ratio),
                                                int(domino_img.shape[1]*scale_ratio)))

            templ_rot = generate_rot_templates(domino_img,
                                               phi - delta_phi,
                                               phi + delta_phi,
                                               num=1)

            templ_rot_rev = generate_rot_templates(domino_img,
                                                   -np.pi + (phi - delta_phi),
                                                   -np.pi + (phi + delta_phi),
                                                   num=1)
            for k in range(len(templ_rot) + len(templ_rot_rev)):
                templ = (templ_rot + templ_rot_rev)[k]
                mask = (templ_mask + templ_mask_rev)[k]
                matching_result = cv.matchTemplate(np.asarray(img, dtype='float32'),
                                                   np.asarray(templ, dtype='float32'),
                                                   method=cv.TM_CCORR_NORMED,
                                                   mask=np.asarray(mask, dtype='float32'))
                if (best_match < np.max(matching_result)):
                    best_phi = phi - np.pi*k
                    best_match = np.max(matching_result)
                    best_matching_result = matching_result
                    best_dom = "%i%i" % (i, j)
                    best_templ = templ
                    (xopt, yopt) = np.unravel_index(np.argmax(best_matching_result, axis=None),
                                                    best_matching_result.shape)
                dom_results.append(['%i%i' % (i, j), best_match])

    return best_dom, (xopt, yopt, best_phi), best_match, best_templ, dom_results


def find_rectangles(img, N, threshold=70000):
    """
    Segments a binary image into single rectangular dominoes and determines their angle and coordinates.

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
        A list of coordinates of the centroids and the rotation angle in radian with respect to the image y-axis
    tile_area : int
        An approximate size of a single domino
    """
    # approx height to width ratio of a domino
    ratio = 1.85
    # labeled_img = np.zeros(bw.shape)
    img = (img == np.max(img))
    total_area = np.sum(img)
    tile_area = total_area/N
    # label all shapes in the image
    (img, m) = label(img)
    # generate a list of rotated dominoes for template matching
    (templ_simple, origin, diag) = create_rect_template(tile_area, ratio=2)
    templates = generate_rot_templates(templ_simple, start=0, end=np.pi, num=30)
    # approx width and height of a domino
    w = int(round(min(np.sqrt(tile_area / ratio), np.sqrt(tile_area * ratio))))
    h = int(round(max(np.sqrt(tile_area / ratio), np.sqrt(tile_area * ratio))))
    coordinates = []

    # # if the number of blobs matches the expected number of dominoes, we're done
    # if m == N:
    #     print("I see %i disconnected dominoes" % m)
    #     for i in range(1, m+1):
    #         blob = np.asarray((img == i), dtype='uint8')
    #         M = cv.moments(blob)
    #         xb = int(round(M['m01'] / M['m00']))
    #         yb = int(round(M['m10'] / M['m00']))
    #         phi = 0.5 * np.arctan2(2 * M['mu11'], (M['mu20'] - M['mu02']))
    #         coordinates.append([xb, yb, phi])
    #     return coordinates, tile_area

    # but it doesn't
    for i in range(1, m+1):
        # separate a blob
        blob = np.asarray((img == i), dtype='uint8')
        blob_area = np.sum(blob)/np.max(blob)
        if np.abs(blob_area - tile_area) < tile_area*0.5:
            M = cv.moments(blob)
            xb = int(round(M['m01'] / M['m00']))
            yb = int(round(M['m10'] / M['m00']))
            phi = np.pi/2 - 0.5*np.arctan2(2*M['mu11'], (M['mu20'] - M['mu02']))
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
                    print("The area of the blob is still larger than the expected area of a domino")
                    print("but I can't find any other rectangles.")
                    break
                # angle of rotation of the template
                phi = np.pi/len(templates)*k
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
            coordinates += approx_coordinates
    return coordinates, tile_area


def preprocessing(img):
    """
    Performs preprosessing on the starting image. Scales image down, sets background pixels to zero, darkens the dots.
    Parameters
    ----------
    img : 3D array
        Starting colorful image from the camera

    Returns
    -------
    bw : uint8 2D array
        A grayscale copy of the image
    binary : uint8 2D array
        A binary image of the siluets of the dominos
    edge : uint8 2D array
        Canny edges of the grayscale image
    img : uint8 2D array
        A scaled down copy of the starting image with darkend colorful pixels
    """
    if img.shape[0] < img.shape[1]:
        img = cv.resize(img, (266, 200))
    else:
        img = cv.resize(img, (200, 266))
    # make colorfull pixels darker
    pixelStd = np.dstack((2*np.std(img, axis=2), )*3)
    img = np.maximum(255*np.zeros(img.shape), img - pixelStd)
    img = np.asarray(img, dtype="uint8")
    # turn image to grayscale
    bw = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # bw = cv.GaussianBlur(bw, (0, 0), 2)
    # set background to black
    (thres, bw) = cv.threshold(bw, 70, 255, cv.THRESH_TOZERO)
    # canny edge
    edge = cv.Canny(bw, 200, 200)
    # binary image
    (thres, binary) = cv.threshold(bw, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    # fill the contours
    # issue: fills background enclosed by dominoes
    binary = fill_holes(binary)

    return bw, binary, edge, img


def cut_segment(coor, seg_shape, img):
    # make dimensions even
    seg_shape = list(seg_shape)
    seg_shape[0] += seg_shape[0] % 2
    seg_shape[1] += seg_shape[1] % 2
    segment = np.zeros(seg_shape)
    img_shape = img.shape
    im_vert = [max(coor[0] - seg_shape[0]//2, 0), min(coor[0] + seg_shape[0]//2, img_shape[0]),
               max(coor[1] - seg_shape[1]//2, 0), min(coor[1] + seg_shape[1]//2, img_shape[1])]
    seg_vert = [max(seg_shape[0]//2 - coor[0], 0), seg_shape[0] - max(seg_shape[0]//2 + coor[0] - img_shape[0], 0),
                max(seg_shape[1]//2 - coor[1], 0), seg_shape[1] - max(seg_shape[1]//2 + coor[1] - img_shape[1], 0)]

    segment[seg_vert[0]:seg_vert[1], seg_vert[2]:seg_vert[3]] = img[im_vert[0]:im_vert[1],
                                                                    im_vert[2]:im_vert[3]]
    segment_coor = [coor[0] - im_vert[0], coor[1] - im_vert[2], coor[2]]
    return segment, segment_coor


def find_dominoes(img, N):
    """
    Recognize dominoes in an image and determine their coordinates, number of points and rotation.
    Parameters
    ----------
    img : 3D array
        A colorful image from the camera
    N : int
        Expected number of dominoes in the imgage. Doesn't have to be exact

    Returns
    -------
    result : list
        A list of lists with each element containing the information about one domino: [domino, (x, y, phi), certainty]
        domino : str
            Has the format "32"
        x, y : int
            Pixel coordinates of the center of the domino
        phi : float
            The angle of a vector pointing from the half with lower number of points to the half with higher number.
            Is defined with respect to the y-axis counterclockwise.
        certainty : float
            The template matching score, can be kinda used to judge how successful the template matching was
    """
    (bw, binary, edge, img) = preprocessing(img)
    coordinates, tile_area = find_rectangles(binary, N)
    result = []
    segmentSize = int(np.sqrt(tile_area) * 2.3)
    for j in range(len(coordinates)):
        coor = coordinates[j]
        segment, segment_coor = cut_segment(coor, (segmentSize, segmentSize), bw)
        segment = np.where(segment > 120, segment, 0)
        best_dom, (xopt, yopt, phi), certainty, templ, dom_result = match_domino(segment, segment_coor, tile_area)
        result.append([best_dom, (xopt + coor[1], yopt + coor[0], phi), certainty])
    return result


if __name__ == '__main__':
    # load images
    folder = "images"
    filename = "uni_test.png"
    images = []
   # for filename in os.listdir(folder):
    for filename in ["uni_test.png", ]:
        img = cv.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)

    bwImg = []
    binaryImg = []
    edgeImg = []
    colorImg = []

    N = 12

    figIm, axIm = plt.subplots(1, 3, dpi=300, sharex=True, sharey=True)

    for i in range(len(images)):
        img = images[i]
        (bw, binary, edge, img) = preprocessing(img)
        bwImg.append(bw)
        binaryImg.append(binary)
        edgeImg.append(edge)
        colorImg.append(img)

    # display one image on a large axis for further examination
    displayID = 0
    N = 1

    axIm[0].imshow(bwImg[displayID], cmap='binary')
    axIm[0].set_title(r'Preprocessing')
    axIm[1].imshow(binaryImg[displayID], cmap='binary')
    axIm[1].set_title('Bitmap image')
    axIm[2].set_title('Final model')

    coordinates, tile_area = find_rectangles(binaryImg[displayID], N)
    axIm[1].plot(np.array(coordinates)[:, 1], np.array(coordinates)[:, 0], 'rx')
    axIm[2].plot(np.array(coordinates)[:, 1], np.array(coordinates)[:, 0], 'rx')
    axIm[2].set_aspect('1')
    for coor in coordinates:
        axIm[1].plot([coor[1] - 20*np.sin(coor[2]), coor[1] + 20*np.sin(coor[2])],
                     [coor[0] - 20*np.cos(coor[2]), coor[0] + 20*np.cos(coor[2])], 'r-')

    for ax in axIm:
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])

    result = find_dominoes(colorImg[displayID], N)

    for i in range(len(result)):
        dom, (x, y, phi), certainty = result[i]
        axIm[2].text(x, y, dom)

    #
    # fig, ax = plt.subplots(1, 3, dpi=300)
    # ax[0].imshow(segment)
    #
    # templIm = np.zeros(segment.shape)
    # templIm[xopt:xopt+templ.shape[0], yopt:yopt+templ.shape[1]] = templ
    #
    # ax[1].imshow(templIm)
    #
    # x = ['%i%i' % (i, j)  for i in range(7) for j in range(i+1) for k in range(2)]
    # figia, axia = plt.subplots(1, 1, dpi=300)
    # try:
    #    axia.lines[0].remove()
    # except IndexError:
    #    pass
    # axia.plot(x, dom_result, 'rs')
    #

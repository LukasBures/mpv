# -*- coding: utf-8 -*-
"""
Created on 10:30 5.10.16

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits:
@version: 1.0.0
"""

import cv2
import numpy as np


def detect_and_describe(img):
    img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    descriptor = cv2.xfeatures2d.SURF_create()
    kps, desc = descriptor.detectAndCompute(img_gs, None)
    return kps, desc


def good_match(match):
    # Store all the good matches as per Lowe's ratio test.
    g_match = []
    for j, k in match:
        if j.distance < (0.7 * k.distance):
            g_match.append(j)
    return g_match


def match(kp1, desc1, kp2, desc2):
    # FLANN_INDEX_KDTREE = 0
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=4)
    # search_params = dict(checks=50)
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # matches = flann.knnMatch(desc1, desc2, k=2)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)

    good = good_match(matches)

    if len(good) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H

    return None


def panorama(imgs, target_size):
    """
    Provede segmentaci pomoci MaxFlow algoritmu.

    :param gsimg: Vstupni sedotonovy obrazek, 255 reprezentuje pozadi a 0 popredi.
    :rtype gsimg: 2D ndarray, uint8

    :return
    :rtype
    """

    k0, d0 = detect_and_describe(imgs[0])
    k1, d1 = detect_and_describe(imgs[1])
    k2, d2 = detect_and_describe(imgs[2])
    # k3, d3 = detect_and_describe(imgs[3])

    # Source -> destination
    H01 = match(k1, d1, k0, d0)
    # k0.extend(k1)
    # d0 = np.concatenate((d0, d1), axis=0)
    H02 = match(k2, d2, k0, d0)
    # k0.extend(k2)
    # d0 = np.concatenate((d0, d2), axis=0)
    #H03 = match(k3, d3, k2, d2)

    result = cv2.warpPerspective(imgs[1], H01, target_size)
    result[0:imgs[0].shape[0], 0:imgs[0].shape[1]] = imgs[0]

    tmp = cv2.warpPerspective(imgs[2], H02, target_size)
    result = cv2.bitwise_or(result, tmp)

    # tmp2 = cv2.warpPerspective(imgs[3], H03, target_size)
    #
    # # for r in range(0, tmp2.shape[0]):
    # #     for c in range(0, tmp2.shape[1]):
    # #         if tmp2[r][c][0] != 0 & tmp2[r][c][1] != 0 & tmp2[r][c][2] != 0:
    # #             result[r][c][:] = tmp2[r][c][:]
    #
    # result = cv2.bitwise_or(result, tmp2)
    return result
    #cv2.imwrite("result.png", result)
    # cv2.namedWindow("Result", 0)
    # cv2.imshow("Result", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

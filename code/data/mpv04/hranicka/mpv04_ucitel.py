# -*- coding: utf-8 -*-
"""
Created on 15:40 18.11.15 

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits: 
@version: 1.0.0
"""

import cv2
import numpy as np


def good_match(match):
    """
    Store all the good matches as per Lowe's ratio test.

    :param match:
    :return:
    """

    goodmatch = []

    for j, k in match:
        if j.distance < (0.7 * k.distance):
            goodmatch.append(j)

    return goodmatch


def calculate_points(l_img, r_img):
    matcher = cv2.BFMatcher()
    sift = cv2.xfeatures2d.SIFT_create()

    l_kp, l_desc = sift.detectAndCompute(l_img, None)
    r_kp, r_desc = sift.detectAndCompute(r_img, None)

    m = matcher.knnMatch(l_desc, r_desc, k=2)
    g_match = good_match(m)

    # if VISUALIZE:
    #     for m in g_match:
    #         (x1, y1) = l_kp[m.queryIdx].pt
    #         (x2, y2) = r_kp[m.trainIdx].pt
    #         cv2.circle(l_img, (int(x1), int(y1)), 3, (0, 0, 255), 2)
    #         cv2.circle(r_img, (int(x2), int(y2)), 3, (0, 0, 255), 2)
    #     cv2.imshow("l_img", l_img)
    #     cv2.imshow("r_img", r_img)

    l_pt = np.float32([l_kp[m.queryIdx].pt for m in g_match]).reshape(-1, 1, 2)
    r_pt = np.float32([r_kp[m.trainIdx].pt for m in g_match]).reshape(-1, 1, 2)

    return l_pt, r_pt


def calculate_f(pts1, pts2):
    if len(pts1) == len(pts2):
        scale1 = 0.0
        scale2 = 0.0
        count = len(pts1)
        m1c = np.zeros((1, 2), dtype=float)
        m2c = np.zeros((1, 2), dtype=float)

        for p1, p2 in zip(pts1, pts2):
            m1c[0, 0] += p1[0][0]
            m1c[0, 1] += p1[0][1]
            m2c[0, 0] += p2[0][0]
            m2c[0, 1] += p2[0][1]

        t = 1.0 / count
        m1c *= t
        m2c *= t

        for p1, p2 in zip(pts1, pts2):
            scale1 += cv2.norm((p1[0][0] - m1c[0, 0], p1[0][1] - m1c[0, 1]))
            scale2 += cv2.norm((p2[0][0] - m2c[0, 0], p2[0][1] - m2c[0, 1]))

        scale1 *= t
        scale2 *= t

        scale1 = np.sqrt(2.0) / scale1
        scale2 = np.sqrt(2.0) / scale2

        aa = np.zeros((len(pts1), 9), dtype=np.float)
        n = 0
        for p1, p2 in zip(pts1, pts2):
            x = (p1[0][0] - m1c[0, 0]) * scale1
            y = (p1[0][1] - m1c[0, 1]) * scale1
            x_ = (p2[0][0] - m2c[0, 0]) * scale2
            y_ = (p2[0][1] - m2c[0, 1]) * scale2
            aa[n, :] = [x * x_, y * x_, x_, x * y_, y * y_, y_, x, y, 1.0]
            n += 1

        u, s, v = np.linalg.svd(aa, full_matrices=True)
        ff = v[8, :].reshape(3, 3)

        u, s, v = np.linalg.svd(ff, full_matrices=True)
        s[2] = 0

        f = np.dot(u, np.dot(np.diag(s), v))

        t1 = np.array([[scale1, 0.0, -scale1 * m1c[0, 0]],
                       [0.0, scale1, -scale1 * m1c[0, 1]],
                       [0.0, 0.0, 1.0]], dtype=float)

        t2 = np.array([[scale2, 0.0, -scale2 * m2c[0, 0]],
                       [0.0, scale2, -scale2 * m2c[0, 1]],
                       [0.0, 0.0, 1.0]], dtype=float)
        print "assa"
        print t1
        print t2

        f0 = np.dot(np.transpose(t2), np.dot(f, t1))
        f0 /= f0[2, 2]

        return f0

    else:
        assert "Sizes aren't equals."


def find_fundamental_matrix(img_left, img_right):
    """

    :param img_left:
    :param img_right:
    :return:
    """
    l_pts, r_pts = calculate_points(img_left, img_right)
    f_my = calculate_f(l_pts, r_pts)

    return f_my

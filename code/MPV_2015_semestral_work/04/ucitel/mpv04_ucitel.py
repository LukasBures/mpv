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

    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    sift = cv2.SIFT()

    l_kp, l_desc = sift.detectAndCompute(l_img, None)
    r_kp, r_desc = sift.detectAndCompute(r_img, None)

    m = matcher.knnMatch(l_desc, r_desc, k=2)
    g_match = good_match(m)

    if VISUALIZE:
        for m in g_match:
            (x1, y1) = l_kp[m.queryIdx].pt
            (x2, y2) = r_kp[m.trainIdx].pt
            cv2.circle(l_img, (int(x1), int(y1)), 3, (0, 0, 255), 2)
            cv2.circle(r_img, (int(x2), int(y2)), 3, (0, 0, 255), 2)
        cv2.imshow("l_img", l_img)
        cv2.imshow("r_img", r_img)

    l_pt = np.float32([l_kp[m.queryIdx].pt for m in g_match]).reshape(-1, 1, 2)
    r_pt = np.float32([r_kp[m.trainIdx].pt for m in g_match]).reshape(-1, 1, 2)

    return l_pt, r_pt


def calculate_essential_matrix(fundamental_mat, intristic_mat_1, intristic_mat_2):
    esential = np.dot(np.dot(np.transpose(intristic_mat_1), fundamental_mat), intristic_mat_2)
    return esential


def decompose_essential_mat(essential_mat):

    w = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    z = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]], dtype=float)

    u, d, v = np.linalg.svd(essential_mat)

    r = np.dot(np.dot(u, w), v)  # prvni zpusob, R = UWV'
    s = np.dot(np.dot(u, z), np.transpose(u))

    b1 = s[2, 1]
    b2 = s[0, 2]
    b3 = s[1, 0]

    t = np.array([b1, b2, b3], dtype=float)

    return r, t


def check_rank_fundamental_mat(fundamental):
    rank = np.linalg.matrix_rank(fundamental)
    if rank == 2:
        print "OK: Fundamental matrix rank =", rank
    else:
        print "Warning: Fundamental matrix rank =", rank


def prepare_s(e):
    e1 = e[0]
    e2 = e[1]
    e3 = e[2]
    s = np.array([[0, -e3, e2], [e3, 0, -e1], [-e2, e1, 0]])
    return s


def calculate_e(fundamental_matrix):
    # Ax=b
    b = np.zeros((3, 1), dtype=np.float)
    # x = np.linalg.solve(fundamental_matrix.T, b)
    u, d, v_transposed = np.linalg.svd(fundamental_matrix.T)
    e = v_transposed[:, 2]

    # Zkouska
    if np.allclose(np.dot(fundamental_matrix.T, e), b):
        print "OK: e"
    else:
        print "Warning: e", np.dot(fundamental_matrix.T, e)
    return e


def prepare_projection_matrices(s, e, fundamental_mat):
    projection1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float)
    projection2 = np.column_stack((np.dot(s, fundamental_mat), e))
    return projection1, projection2


def draw_matches(img1, kp1, img2, kp2, match):
    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')

    # Place the first image to the left
    out[0:rows1, 0:cols1, :] = img1

    # Place the next image to the right of it
    out[0:rows2, cols1:cols1 + cols2, :] = img2

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in match:
        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1), int(y1)), 10, (255, 0, 0), -1)
        cv2.circle(out, (int(x2) + cols1, int(y2)), 10, (0, 255, 0), -1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1), int(y1)),
                 (int(x2) + cols1, int(y2)), (0, 0, 255), 3)

    return out


def calculate(img_left, img_right):
    global VISUALIZE
    VISUALIZE = True

    l_pts, r_pts = calculate_points(img_left, img_right)

    fundamental_mat, used_pos = cv2.findFundamentalMat(l_pts, r_pts, cv2.cv.CV_FM_RANSAC, 3, 0.99)
    print "Fundamental matrix ="
    print fundamental_mat
    check_rank_fundamental_mat(fundamental_mat)

    e = calculate_e(fundamental_mat)
    print "e =", e

    s = prepare_s(e)
    print "s =", s

    p1, p2 = prepare_projection_matrices(s, e, fundamental_mat)
    print "p1 =", p1
    print "p2 =", p2

    l_pts = l_pts[:, -1, :]
    r_pts = r_pts[:, -1, :]

    xyz_homo_t = cv2.triangulatePoints(p1, p2, l_pts.T, r_pts.T)
    xyz_homo = xyz_homo_t.T

    x = xyz_homo[:, 0] / xyz_homo[:, 3]
    y = xyz_homo[:, 1] / xyz_homo[:, 3]
    z = xyz_homo[:, 2] / xyz_homo[:, 3]

    x_out = []
    y_out = []
    z_out = []
    used_pos = used_pos[:, -1]
    for ix, iy, iz, pos in zip(x, y, z, used_pos):
        if pos:
            x_out.append(ix)
            y_out.append(iy)
            z_out.append(iz)

    x_out = np.array(x_out)
    y_out = np.array(y_out)
    z_out = np.array(z_out)

    xyz = np.column_stack((x_out, y_out, z_out))

    return xyz

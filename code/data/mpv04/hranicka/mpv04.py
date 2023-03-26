# -*- coding: utf-8 -*-
"""
Computer Vision - Semester Project 4
@author: Bc. Jan Hranicka
@email: jhranick@students.zcu.cz
@version: 1.0
"""

import cv2
import numpy as np
from os.path import join
import logging

# Logging utility initialization
logger = logging.getLogger('mpv04')
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

logger.addHandler(ch)


def normalize(pts):
    """
    Method for points normalization, return projection matrix
    :param pts: Picture points
    :return: tuple: normalized points, projection matrix H
    """
    means = np.mean(pts, axis=0)
    stds = np.std(pts, axis=0)

    newpts = np.zeros((len(pts), 2))
    newpts[:, 0] = pts[:, 0] - means[0]
    newpts[:, 1] = pts[:, 1] - means[1]
    #newpts[:, 2] = 1.0

    dist = np.sum(np.sqrt(np.power(newpts, 2)), axis=1)
    meandist = np.mean(dist)

    scale = np.sqrt(2)/meandist

    H = np.array([
        [scale, 0, -scale*means[0]],
        [0, scale, -scale*means[1]],
        [0, 0, 1]
    ])

    # H = np.array([
    #     [ 1.0/means[0], 0.0, -means[0]/stds[0]],
    #     [0.0, means[1], -means[1]/stds[1]],
    #     [0.0, 0.0, 1.0]
    # ])

    newpts = []
    for pt in pts:
        newpts.append(np.dot(H, pt))
    newpts = np.asarray(newpts)

    print H

    return newpts, H


def find_fundamental_matrix(img_left, img_right):
    """
        Provede vypocet fundamentalni matice.
    :param img_left: Vstupni levy obrazek.
    :rtype img_left: 2D ndarray, uint8
    :param img_right: Vstupni pravy obrazek.
    :rtype img_right: 2D ndarray, uint8
    :return f: Vypoctena fundamentalni matice velikosti 3x3.
    :rtype f: 2D ndarray, float
    """

    # Find SIFT keypoints
    sift = cv2.xfeatures2d.SIFT_create()
    kps_l, descs_l = sift.detectAndCompute(img_left, None)
    kps_r, descs_r = sift.detectAndCompute(img_right, None)

    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descs_l, descs_r, k=2)

    # Lowe's ratio test
    gamma = 0.7
    good_matches = []
    for m, n in matches:
        if m.distance < gamma * n.distance:
            good_matches.append(m)

    PT_l = np.float32([kps_l[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    PT_r = np.float32([kps_r[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    PT_l = np.squeeze(PT_l)
    PT_r = np.squeeze(PT_r)
    pom = np.ones((PT_l.shape[0], PT_l.shape[1] + 1))
    pom[:,:-1] = PT_l
    PT_l = pom

    pom = np.ones((PT_r.shape[0], PT_r.shape[1] + 1))
    pom[:, :-1] = PT_r
    PT_r = pom

    # cv2F, _ = cv2.findFundamentalMat(PT_l, PT_r, cv2.FM_8POINT)
    # print cv2F
    # print "-------------------------------------"

    # Do normalization of coordinates
    # To make the algorithm numerical stable, normalize image coordinates so that RMS equals sqrt(2)
    [PT_l, H_l] = normalize(PT_l)
    [PT_r, H_r] = normalize(PT_r)

    # Find the fundamental matrix
    # Build the constraint matrix from all image points
    listA = list()
    for k in range(len(PT_l)):
        #listA.append(np.array([i*j for i in PT_r[0,:] for j in PT_l[0,:]]))  # prvni nastrel
        listA.append(np.array([i*j for i in PT_r[k,:] for j in PT_l[k,:]]))
    A = np.asarray(listA)
    U, S, V = np.linalg.svd(A, full_matrices=True)  # Compute SVD


    # TODO: spatny vyber z matice V
    # F = np.transpose(V[:,-1].reshape(3,3))          # Use the 9th column as F matrix (not fundamental)
    F = np.transpose(V[8, :].reshape(3, 3))  # Use the 9th column as F matrix (not fundamental)


    #F = np.transpose(V[-1].reshape(3, 3))
    U, S, V = np.linalg.svd(F, full_matrices=True)  # Compute SVD of new F matrix

    S[2]=0
    diag = np.diag(S)
    # TODO - asi spatny tvar
    # F = np.dot(np.dot(U, diag), np.transpose(V))
    F = np.dot(U, np.dot(diag, V))


    # print "Found fundamental matrix"

    # Do denormalization of coordinates
    # print "Denormalized"
    F = np.dot(np.transpose(H_l), np.dot(F, H_r))
    # TODO - chybi normalizace
    F /= F[2,2]
    # print F

    return F

if __name__ == '__main__':
    # Loading images
    imagesPath = r"D:\OneDrive\Faculty of Applied Science\Postgraduate Master Studies\2016-2017\MPV\Semester Projects Repository\SP4\img"
    img_l = join(imagesPath, "l1.jpg")
    img_r = join(imagesPath, "r1.jpg")

    img_l = cv2.imread(img_l, cv2.IMREAD_GRAYSCALE)
    img_r = cv2.imread(img_r, cv2.IMREAD_GRAYSCALE)

    F = find_fundamental_matrix(img_l, img_r)

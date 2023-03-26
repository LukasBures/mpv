# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 12:58:48 2014

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits: 
@version: 2.0.0

Revision Note:
2.0.0 - 5.12.2016 - Updated for OpenCV 3.1.0 version
"""

import cv2
import numpy as np


def draw_matches(img1, kp1, img2, kp2, matches):
    # Create a new output image that concatenates the two images together (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype="uint8")

    # Place the first image to the left
    out[0:rows1, 0:cols1, :] = img1

    # Place the next image to the right of it
    out[0:rows2, cols1:cols1 + cols2, :] = img2

    # For each pair of points we have between both images draw circles, then connect a line between them.
    for mat in matches:
        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        B = np.random.randint(0, 256)
        G = np.random.randint(0, 256)
        R = np.random.randint(0, 256)

        cv2.circle(out, (int(x1), int(y1)), 3, (B, G, R), 2)
        cv2.circle(out, (int(x2) + cols1, int(y2)), 3, (B, G, R), 2)

        cv2.line(out, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (B, G, R), 1, cv2.LINE_AA)

    return out


def draw_lines(img, lines, bgr):
    r, c, _ = img.shape

    n = 0
    for l in lines:
        l = np.squeeze(l)
        pt1 = np.array([0, l[2] / - l[1]], np.float32)
        pt2 = np.array([c, ((l[0] * c) / - l[1]) + (l[2] / - l[1])], np.float32)
        pt1 = np.int32(pt1)
        pt2 = np.int32(pt2)
        cv2.line(img, (pt1[0], pt1[1]), (pt2[0], pt2[1]), bgr[n], 3, cv2.LINE_AA)
        n += 1


if __name__ == '__main__':

    print "Load Images ...",
    Img_l = cv2.imread("./img/l1.jpg", cv2.IMREAD_COLOR)
    Img_r = cv2.imread("./img/r1.jpg", cv2.IMREAD_COLOR)
    print "OK"

    # ------------------------------------------------------------------------------------------------------------------
    # Convert to grayscale images.
    print "Convert to grayscale images ...",
    ImgGS_l = cv2.cvtColor(Img_l, cv2.COLOR_RGB2GRAY)
    ImgGS_r = cv2.cvtColor(Img_r, cv2.COLOR_RGB2GRAY)
    print "OK"

    # ------------------------------------------------------------------------------------------------------------------
    # Compute SIFT points.
    print "Compute SIFT points ...",
    sift = cv2.xfeatures2d.SIFT_create()
    KP_l, Des_l = sift.detectAndCompute(ImgGS_l, None)
    KP_r, Des_r = sift.detectAndCompute(ImgGS_r, None)
    print "OK"

    # ------------------------------------------------------------------------------------------------------------------
    # Match SIFT and prepare 2D points.
    print "Match SIFT and prepare 2D points ...",
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    Matches = matcher.knnMatch(Des_l, Des_r, k=2)

    # Apply ratio test.
    GoodMatches = []
    for m, n in Matches:
        if m.distance < 0.6 * n.distance:
            GoodMatches.append(m)

    PT_l = np.float32([KP_l[m.queryIdx].pt for m in GoodMatches]).reshape(-1, 1, 2)
    PT_r = np.float32([KP_r[m.trainIdx].pt for m in GoodMatches]).reshape(-1, 1, 2)
    print "OK"

    # ------------------------------------------------------------------------------------------------------------------
    # Visualization.
    print "Visualization of keypoints matches ...",
    ImgSIFT_l = cv2.drawKeypoints(ImgGS_l, KP_l[:], None, (0, 0, 255),
                                  cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    ImgSIFT_r = cv2.drawKeypoints(ImgGS_r, KP_r[:], None, (0, 0, 255),
                                  cv2.DRAW_MATCHES_FLAGS_DEFAULT)

    cv2.namedWindow("SIFT_l", 0)
    cv2.namedWindow("SIFT_r", 0)
    cv2.imshow("SIFT_l", ImgSIFT_l)
    cv2.imshow("SIFT_r", ImgSIFT_r)

    matchesImg = draw_matches(Img_l, KP_l, Img_r, KP_r, GoodMatches)

    cv2.namedWindow("matchesImg", 0)
    cv2.imshow("matchesImg", matchesImg)
    print "OK"

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ------------------------------------------------------------------------------------------------------------------
    # Find Fundamental matrix.
    print "Project points in to the second image ...",
    F, mask = cv2.findFundamentalMat(PT_l, PT_r, cv2.FM_LMEDS)

    # Chose just points which were used for calculation.
    PT_l = PT_l[mask.ravel() == 1]
    PT_r = PT_r[mask.ravel() == 1]

    nPt = 15
    colors = []
    for i in range(nPt):
        B = np.random.randint(0, 256)
        G = np.random.randint(0, 256)
        R = np.random.randint(0, 256)
        colors.append((B, G, R))

    lines1 = cv2.computeCorrespondEpilines(PT_l, 1, F)
    draw_lines(Img_r, lines1[:nPt], colors)

    lines2 = cv2.computeCorrespondEpilines(PT_r, 2, F)
    draw_lines(Img_l, lines2[:nPt], colors)

    for i in range(nPt):
        cv2.circle(Img_l, (int(PT_l[i, 0][0]), int(PT_l[i, 0][1])), 10, colors[i], -1)
        cv2.circle(Img_r, (int(PT_r[i, 0][0]), int(PT_r[i, 0][1])), 10, colors[i], -1)

    cv2.namedWindow("Corners l", 0)
    cv2.namedWindow("Corners r", 0)
    cv2.imshow("Corners l", Img_l)
    cv2.imshow("Corners r", Img_r)
    print "OK"

    cv2.waitKey(0)
    cv2.destroyAllWindows()

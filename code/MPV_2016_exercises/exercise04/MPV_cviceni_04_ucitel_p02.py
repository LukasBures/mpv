# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 15:52:49 2014

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits: Ing. Marek Hruz, Ph.D.
@version: 2.0.0

Revision Note:
2.0.0 - 13.10.2016 - Updated for OpenCV 3.1.0 version
"""

import cv2
import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# Create one image from two input images.
def side_by_side(img1, img2):
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

    return out


# ----------------------------------------------------------------------------------------------------------------------
# It draws KP matches.
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

    # For each pair of points we have between both images draw circles, then connect a line between them
    for mat in match:
        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        cv2.circle(out, (int(x1), int(y1)), 10, (255, 0, 0), -1)
        cv2.circle(out, (int(x2) + cols1, int(y2)), 10, (0, 255, 0), -1)

        # Draw a line in between the two points
        cv2.line(out, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (0, 0, 255), 3)

    return out


def good_match(match):

    # Store all the good matches as per Lowe's ratio test.
    goodmatch = []
    for j, k in match:
        if j.distance < (0.7 * k.distance):
            goodmatch.append(j)

    return goodmatch


# ----------------------------------------------------------------------------------------------------------------------
# SIFT, SURF, ORB, AKAZE detectors, descriptors, matcher, find homography
if __name__ == '__main__':
    # mode: 1 = SIFT, 2 = SURF, 3 = ORB, 4 = AKAZE
    mode = 1

    cv2.namedWindow("Matched Keypoints", cv2.WINDOW_NORMAL)

    # Load color image and convert it in to the grayscale color
    Template = cv2.imread("./img/Stock_84.png", cv2.IMREAD_COLOR)
    TemplateGS = cv2.cvtColor(Template, cv2.COLOR_RGB2GRAY)
    cap = cv2.VideoCapture("./img/trimmed.avi")

    sift = []
    surf = []
    orb = []
    akaze = []

    if mode == 1:
        print "SIFT mode"
        # FLANN_INDEX_KDTREE = 0
        # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        # search_params = dict(checks=50)
        # matcher = cv2.FlannBasedMatcher(index_params, search_params)
        # OpenCV 2.7.X
        # sift = cv2.SIFT(0, 3, 0.04, 10, 1.6)
        # OpenCV 3.1.0
        matcher = cv2.BFMatcher()
        sift = cv2.xfeatures2d.SIFT_create(0, 3, 0.04, 10, 1.6)
        kpTemplate, desTemplate = sift.detectAndCompute(TemplateGS, None)

    elif mode == 2:
        print "SURF mode"
        # FLANN_INDEX_KDTREE = 0
        # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        # search_params = dict(checks=50)
        # matcher = cv2.FlannBasedMatcher(index_params, search_params)
        # OpenCV 2.7.X
        # surf = cv2.SURF(400, 3, 3, 1, 1)
        # OpenCV 3.1.0
        matcher = cv2.BFMatcher()
        surf = cv2.xfeatures2d.SURF_create(400, 3, 3, 1, 1)
        kpTemplate, desTemplate = surf.detectAndCompute(TemplateGS, None)

    elif mode == 3:
        print "ORB mode"
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Vytvoreni ORB objektu.
        # OpenCV 2.7.X
        # orb = cv2.ORB(nORB, 1.2, 8, 31, 0, 2, cv2.ORB_HARRIS_SCORE, 31)
        # OpenCV 3.1.0
        orb = cv2.ORB_create(500, 1.2, 8, 31, 0, 2, cv2.ORB_HARRIS_SCORE, 31)
        kpTemplate, desTemplate = orb.detectAndCompute(TemplateGS, None)

    elif mode == 4:
        print "AKAZE"
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        akaze = cv2.AKAZE_create()
        kpTemplate, desTemplate = akaze.detectAndCompute(TemplateGS, None)

    else:
        print "Unknown mode."

    # Main loop
    while True:

        # Capture frame-by-frame
        ret, Video = cap.read()
        # Video = cv2.flip(Video, 0)

        # Exit if video ends
        if not ret:
            break

        # Convert image in to the grayscale color
        VideoGS = cv2.cvtColor(Video, cv2.COLOR_RGB2GRAY)

        # Detect keypoints and compute descriptors
        if mode == 1:
            kpVideoImg, desVideo = sift.detectAndCompute(VideoGS, None)
            # if desVideo is not None:
            #     matches = matcher.knnMatch(desTemplate, desVideo, k=2)
            #     goodMatch = good_match(matches)

        elif mode == 2:
            kpVideoImg, desVideo = surf.detectAndCompute(VideoGS, None)
            # if desVideo is not None:
            #     matches = matcher.knnMatch(desTemplate, desVideo, k=2)
            #     goodMatch = good_match(matches)

        elif mode == 3:
            kpVideoImg, desVideo = orb.detectAndCompute(VideoGS, None)
        elif mode == 4:
            kpVideoImg, desVideo = akaze.detectAndCompute(VideoGS, None)

        else:
            "Unknown mode."
            break

        if desVideo is not None:
            matches = matcher.match(desTemplate, desVideo)
            # matches = matcher.knnMatch(desTemplate, desVideo, k=2)
            # goodMatch = good_match(matches)
            matches = sorted(matches, key=lambda val: val.distance)
            goodMatch = matches[:np.int(len(matches) / 4)]

            print "Number of good matches:", len(goodMatch)

        if len(goodMatch) > 4:
            ptsTemplate = np.float32([kpTemplate[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
            ptsVideoImg = np.float32([kpVideoImg[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(ptsTemplate, ptsVideoImg, cv2.RANSAC)
            matchesMask = mask.ravel().tolist()

            h, w, _ = Template.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            # dst = cv2.warpPerspective(pts, H, (h, w))

            dst = cv2.perspectiveTransform(pts, H)

            # Draw template border
            cv2.polylines(Video, [np.int32(dst)], True, (0, 255, 0), 5)

            # Draw keypoits match
            # OpenCV 2.7.X
            # matchesImg = draw_matches(Template, kpTemplate, Video, kpVideoImg, goodMatch)
            # OpenCV 3.1.0
            # cv2.drawMatches()
            out = cv2.drawMatches(Template, kpTemplate, Video, kpVideoImg, goodMatch, Video)

        # elif kpVideoImg:
        #     matchesImg = draw_matches(Template, kpTemplate, Video, kpVideoImg, goodMatch)
        # else:

        # matchesImg = side_by_side(Template, Video)

        # Draw results
        # out = cv2.transpose(out)

        out = cv2.resize(out, (640, 480))

        cv2.imshow("Matched Keypoints", out)

        # ESC key
        if cv2.waitKey(1) == 27:
            break

    # Destroy all windows
    cv2.destroyAllWindows()

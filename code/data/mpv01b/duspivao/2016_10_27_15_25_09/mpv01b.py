# Author: Ondrej Duspiva

import cv2
import numpy as np

# just for test
# imgs = [cv2.imread("../sampleData/0.png"), cv2.imread("../sampleData/1.png"), cv2.imread("../sampleData/2.png")]
# inSize = (1920, 1080)
# # imgs = [cv2.imread("C:/Users/NTB/Pictures/test/0.png"), cv2.imread("C:/Users/NTB/Pictures/test/1.png"), cv2.imread("C:/Users/NTB/Pictures/test/2.png"), cv2.imread("C:/Users/NTB/Pictures/test/3.png"), cv2.imread("C:/Users/NTB/Pictures/test/4.png"), cv2.imread("C:/Users/NTB/Pictures/test/5.png"), cv2.imread("C:/Users/NTB/Pictures/test/6.png")]
# # inSize = (730, 489)
# # imgs = [cv2.imread("C:/Users/NTB/Pictures/test/0.png"), cv2.imread("C:/Users/NTB/Pictures/test/3.png"), cv2.imread("C:/Users/NTB/Pictures/test/1.png"), cv2.imread("C:/Users/NTB/Pictures/test/2.png"), cv2.imread("C:/Users/NTB/Pictures/test/4.png")]
# # inSize = (730, 489)

# just for test

def stich(img1, img2, img1Gs, img2Gs, matches, key_points1, key_points2, size):
    pts1 = np.empty((len(matches), 2))
    pts2 = np.empty((len(matches), 2))

    for i in range(len(matches)):
        (x, y) = key_points1[matches[i].queryIdx].pt
        pts1[i, 0] = round(x, 0)
        pts1[i, 1] = round(y, 0)
        (x1, y1) = key_points2[matches[i].trainIdx].pt
        pts2[i, 0] = round(x1, 0)
        pts2[i, 1] = round(y1, 0)
    h, _ = cv2.findHomography(pts2, pts1)
    imgF = cv2.warpPerspective(img2, h, size)
    imgFGs = cv2.warpPerspective(img2Gs, h, size)
    # cv2.imshow("Matches", cv2.resize(img1, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC))

    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if i < img1.shape[0] and j < img1.shape[1]:
                if img1Gs[i, j] != 0 and imgFGs[i, j] == 0:
                    imgF[i, j] = img1[i, j]
    return imgF

def panorama(imgs, size):
    imgs_gs = list()
    key_points = list()
    descriptors = list()
    matcher = cv2.BFMatcher()
    # sift = cv2.xfeatures2d.SIFT_create()
    sift = cv2.xfeatures2d.SURF_create()

    for i in range(len(imgs)):
        image = imgs[i]
        imTemp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imgs_gs.append(imTemp)
        temp = sift.detectAndCompute(imgs_gs[i], None)
        key_points.append(temp[0])
        descriptors.append(temp[1])
    pocet_pruchodu = 1
    while len(imgs) > 1:
        tempImg = imgs[0]
        tempImgGs = imgs_gs[0]
        tempKeyPoints = key_points[0]
        for i in range(1, len(imgs_gs)):

            matches = matcher.knnMatch(descriptors[0], descriptors[i], k=2)
            good_matches = list()
            for m, n in matches:
                if m.distance < 0.1 * n.distance:
                    good_matches.append(m)
            matches = good_matches
            pocet_pruchodu += 1
            if len(matches) > 0:
                img = stich(tempImg, imgs[i], tempImgGs, imgs_gs[i], matches, tempKeyPoints, key_points[i], size)
                imgs[0] = img
                imgs.pop(i)
                imgs_gs[0] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                imgs_gs.pop(i)
                temp2 = sift.detectAndCompute(img, None)
                key_points[0] = temp2[0]
                descriptors[0] = temp2[1]
                key_points.pop(i)
                descriptors.pop(i)
                break
    # cv2.imshow("MatchesFinal", cv2.resize(imgs[0], None, fx=0.75, fy=0.75, interpolation=cv2.INTER_CUBIC))
    # print imgs[0].shape
    return imgs[0]
# # #
# panorama(imgs, inSize)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

# img0 =  cv2.imread("0.png", cv2.IMREAD_COLOR)
# img1 =  cv2.imread("1.png", cv2.IMREAD_COLOR)
# img2 =  cv2.imread("2.png", cv2.IMREAD_COLOR)
# img3 =  cv2.imread("3.png", cv2.IMREAD_COLOR)
# sizeWindow = [1920,1080]
#
# listPics = []
# listPics.append(img0)
# listPics.append(img1)
# listPics.append(img2)
# listPics.append(img3)

surf = cv2.xfeatures2d.SIFT_create()

def matches(des1,des2):

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.4 * n.distance:
            good.append([m])
    return good

def panorama(imgs, target_size):

    img0Big = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    img0 = imgs[0]
    #nakopirovani prvniho obrazku do leveho horniho rohu s velikosti target_size
    for i, val in enumerate(img0Big):
        for j, val in enumerate(img0Big[i]):
            if i < len(imgs[0]):
                if j < len(img0[i]):
                    img0Big[i][j] = img0[i][j]


    isDone = np.empty(len(imgs)-1, np.int8)
    iDxImg = np.empty(len(imgs)-1, np.int8)

    for i in range(0, len(imgs)-1):
        isDone[i] = 0
        iDxImg[i] = i+1

    tmp = 1

    while(sum(isDone) < len(imgs)-1):

        if isDone[tmp-1] == 0:

            #ziskani keypoints a descriptoru
            key1, des1 = surf.detectAndCompute(img0Big.astype(np.uint8), None)
            key2, des2 = surf.detectAndCompute(imgs[tmp], None)

            #filtrovani jen dobrych matchu
            good = matches(des1, des2)
            print len(good)

            if len(good) > 0:

                #nove prazdne numpy pole pro vypocet homografie
                x = np.empty((len(good), 2))
                y = np.empty((len(good), 2))

                for i in range(0, len(good) - 1):
                    x[i] = key1[good[i][0].queryIdx].pt
                    y[i] = key2[good[i][0].trainIdx].pt

                #nalezeni matice homografie
                h, _ = cv2.findHomography(y, x,cv2.RANSAC, 3.0)

                imgF = cv2.warpPerspective(imgs[tmp], h, (1920, 1080))

                # spojeni obrazku dohromady
                # for i, val in enumerate(img0Big):
                #     for j, val in enumerate(img0Big[i]):
                #         if all(img0Big[i][j]) == 0:
                #             img0Big[i][j] = imgF[i][j]

                odecet = cv2.subtract(img0Big, imgF)

                img0Big = cv2.add(odecet,imgF)

                # iii = np.array([])
                # img3 = cv2.drawMatchesKnn(img0,key1,img2,key2,good,iii,flags=2)
                # plt.imshow(img3, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
                # plt.show()

                # plt.imshow(cv2.cvtColor(img0Big, cv2.COLOR_BGR2RGB))
                # plt.show()

                isDone[tmp - 1] = 1

        if tmp == len(imgs)-1:
            tmp = 0
        tmp += 1

    return img0Big

#stitch(listPics, sizeWindow)
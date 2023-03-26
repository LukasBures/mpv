# -*- coding: utf-8 -*-
"""
Created on 13:39 6.10.15 

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits: 
@version: 1.0.0
"""

import mpv01c
import cv2
import numpy as np
import os

if __name__ == '__main__':

    path = "sampleData/src/43.jpg"

    # for filename in os.listdir(path):
    gsimg = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    print len(gsimg)
    print gsimg.shape

    block_size = 3
    ksize = 3
    k = 0.04

    corner_img1 = mpv01c.harris(gsimg, block_size, ksize, k)
    corner_img1 = np.asarray(corner_img1)

    corner_img2 = cv2.cornerHarris(gsimg, block_size, ksize, k)
    cv2.normalize(corner_img2, corner_img2, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    diff = np.abs(corner_img1 - round(corner_img2))
    sum = np.sum(np.sum(diff))
    print sum




    # cv2.imshow("res", corner_img1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # corner_img2 = cv2.cornerHarris(gsimg, block_size, ksize, k, borderType=cv2.BORDER_DEFAULT)
    #
    # norm_img1 = cv2.normalize(corner_img1, alpha=0.0, beta=255.0, norm_type=cv2.NORM_MINMAX)
    # norm_img2 = cv2.normalize(corner_img2, alpha=0.0, beta=255.0, norm_type=cv2.NORM_MINMAX)
    #
    # norm_img1 = np.uint8(norm_img1)
    # norm_img2 = np.uint8(norm_img2)
    #
    # # np.allclose()
    # if np.array_equal(norm_img1, norm_img2):
    #     print "OK"
    # else:
    #     e = np.sum(np.sum(np.abs(np.subtract(norm_img1, norm_img2))))
    #     print "Error=", e
    #     os.remove(path + filename)
    #
    #     # array_np = np.asarray(np.abs(np.subtract(norm_img1, norm_img2)))
    #     # idx = array_np > 0
    #     # array_np[idx] = 255
    #     #
    #     # cv2.imshow("as", np.uint8(array_np))
    #     # cv2.imshow("My Harris", norm_img1)
    #     # cv2.imshow("OpenCV Harris", norm_img2)
    #     # cv2.imshow("subst", np.abs(np.subtract(norm_img1, norm_img2)))
    #     #
    #     # cv2.waitKey(0)
    #
    # print " "

# -*- coding: utf-8 -*-
"""
Created on 13:39 6.10.15 

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits: 
@version: 1.0.0
"""

import mpv01c_ucitel
import cv2
import numpy as np

if __name__ == '__main__':

    gsimg = cv2.imread("../sampleData/src/000.jpg", cv2.CV_LOAD_IMAGE_GRAYSCALE)

    block_size = 3
    ksize = 3
    k = 0.04

    corner_img1 = mpv01c_ucitel.harris(gsimg, block_size, ksize, k)
    corner_img2 = cv2.cornerHarris(gsimg, block_size, ksize, k, borderType=cv2.BORDER_DEFAULT)

    norm_img1 = cv2.normalize(corner_img1, alpha=0.0, beta=255.0, norm_type=cv2.NORM_MINMAX)
    norm_img2 = cv2.normalize(corner_img2, alpha=0.0, beta=255.0, norm_type=cv2.NORM_MINMAX)

    norm_img1 = np.uint8(norm_img1)
    norm_img2 = np.uint8(norm_img2)

    # np.allclose()
    if np.array_equal(norm_img1, norm_img2):
        print "OK"
        e = np.sum(np.sum(np.abs(np.subtract(norm_img1, norm_img2))))
        print "Error=", e
    else:
        e = np.sum(np.sum(np.abs(np.subtract(norm_img1, norm_img2))))
        print "Error=", e

        array_np = np.asarray(np.abs(np.subtract(norm_img1, norm_img2)))
        idx = array_np > 0
        array_np[idx] = 255

        cv2.imshow("as", np.uint8(array_np))
        cv2.imshow("My Harris", norm_img1)
        cv2.imshow("OpenCV Harris", norm_img2)
        cv2.imshow("subst", np.abs(np.subtract(norm_img1, norm_img2)))

        cv2.waitKey(0)


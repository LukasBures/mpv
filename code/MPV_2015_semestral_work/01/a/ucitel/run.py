# -*- coding: utf-8 -*-
"""
Created on Wed Oct 01 17:13:56 2014

@author: Lukas Bures
"""


import mpv01a_ucitel
import cv2

if __name__ == '__main__':

    gsimg = cv2.imread("../data/syntheticImg_0.jpg", cv2.CV_LOAD_IMAGE_GRAYSCALE)

    th1 = mpv01a_ucitel.otsu(gsimg)
    th2, _ = cv2.threshold(gsimg, 0, 255, cv2.THRESH_OTSU)

    print "Ucitel=", th1
    print "Otsu cv2=", th2

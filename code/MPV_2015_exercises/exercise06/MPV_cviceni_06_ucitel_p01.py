# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 18:03:11 2014

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@version: 2.0.0
"""

import cv2

# ------------------------------------------------------------------------------
# Priklad 1: Background Subtraction OpenCV implementace.
# ------------------------------------------------------------------------------
if __name__ == '__main__':

    # ./img/Megamind.avi
    cap = cv2.VideoCapture("./img/video.avi")
    cv2.namedWindow("Original Video", 1)
    cv2.namedWindow("Foreground Mask by MOG", 1)
    cv2.namedWindow("Foreground Mask by MOG2", 1)
    # cv2.namedWindow("Foreground Mask by GMG", 1)
    
    BS1 = cv2.BackgroundSubtractorMOG()
    BS2 = cv2.BackgroundSubtractorMOG2()

    # GMG is implemented in C++
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # BS3 = cv2.createBackgroundSubtractorGMG()

    while True:
        ret, Img = cap.read()       

        if ret:
            cv2.imshow("Original Video", Img)

            fgmask1 = BS1.apply(Img)
            cv2.imshow('Foreground Mask by MOG', fgmask1)

            fgmask2 = BS2.apply(Img)
            cv2.imshow('Foreground Mask by MOG2', fgmask2)

            # fgmask3 = BS3.apply(Img)
            # fgmask3 = cv2.morphologyEx(fgmask3, cv2.MORPH_OPEN, kernel)
            # cv2.imshow('Foreground Mask by GMG', fgmask3)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

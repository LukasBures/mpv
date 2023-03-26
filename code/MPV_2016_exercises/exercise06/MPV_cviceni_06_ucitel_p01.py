# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 18:03:11 2014

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@version: 3.0.0

Revision Note:
3.0.0 - 24.10.2016 - Updated for OpenCV 3.1.0 version
"""

import cv2

# ----------------------------------------------------------------------------------------------------------------------
# Priklad 1: Background Subtraction OpenCV implementace.
if __name__ == '__main__':
    # ./img/Megamind.avi
    cap = cv2.VideoCapture("./img/Megamind.avi")
    cv2.namedWindow("Original Video", 1)
    cv2.namedWindow("Foreground Mask by BGS KNN", 1)
    cv2.namedWindow("Foreground Mask by BGS MOG2", 1)

    BS1 = cv2.createBackgroundSubtractorKNN()
    BS2 = cv2.createBackgroundSubtractorMOG2()

    while True:
        ret, Img = cap.read()       

        if ret:
            fgmask1 = BS1.apply(Img)
            cv2.imshow('Foreground Mask by BGS KNN', fgmask1)

            fgmask2 = BS2.apply(Img)
            cv2.imshow('Foreground Mask by BGS MOG2', fgmask2)

            cv2.imshow("Original Video", Img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

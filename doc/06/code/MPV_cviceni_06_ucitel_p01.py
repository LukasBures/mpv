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
    
    BS1 = cv2.BackgroundSubtractorMOG()
    BS2 = cv2.BackgroundSubtractorMOG2()

    while True:
        ret, Img = cap.read()       

        if ret:
            cv2.imshow("Original Video", Img)

            fgmask1 = BS1.apply(Img)
            cv2.imshow('Foreground Mask by MOG', fgmask1)

            fgmask2 = BS2.apply(Img)
            cv2.imshow('Foreground Mask by MOG2', fgmask2)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

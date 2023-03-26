# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 09:27:14 2014
"""

# Import modulu.
import cv2

def histogram(img):
    img =  cv2.cvtColor(img, cv2.cv.CV_RGB2GRAY)
    hist = cv2.calcHist([img], [0], None, [256], [0,256])
    return hist

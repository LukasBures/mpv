# -*- coding: utf-8 -*-
"""
Created on Wed Oct 08 12:59:17 2014

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@version: 2.0.0
"""


# Import modulu.
import cv2 
import numpy as np 
import copy


# ------------------------------------------------------------------------------
# Priklad 1: Harris Corner Detector.
# ------------------------------------------------------------------------------

# Nacteni barevneho obrazku a prevedeni do odstinu sedi.
Img = cv2.imread("./img/homer.jpg", cv2.IMREAD_COLOR)
ImgGS = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)

# Vykresleni nacteneho obrazku.
cv2.namedWindow("Original Image", cv2.WINDOW_AUTOSIZE)
cv2.moveWindow("Original Image", 100, 100)
cv2.imshow("Original Image", Img)    
cv2.waitKey(0)

cv2.destroyAllWindows()

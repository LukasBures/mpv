# -*- coding: utf-8 -*-
"""
Created on Wed Oct 08 12:59:17 2014

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits: Ing. Eva Turnerova
@version: 1.0.0
"""


# Import OpenCV modulu.
import cv2 

# Import modulu pro pocitani s Pythonem.
import numpy as np 


# Nacteni barevneho obrazku a prevedeni do odstinu sedi
Img = cv2.imread("lena.jpg", cv2.IMREAD_COLOR)
ImgGS = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)

# Vykresleni
cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
cv2.imshow("Original Image", Img)    
cv2.waitKey(1)


ImgGS = np.float32(ImgGS)
blockSize = 2;
apertureSize = 3;
k = 0.04;
HarrisImg = cv2.cornerHarris(ImgGS, blockSize, apertureSize, k);

# Vykresleni
cv2.namedWindow("Harris Corners", cv2.WINDOW_NORMAL)
cv2.imshow("Harris Corners", HarrisImg)    
cv2.waitKey(1)

# Normalizace
HarrisImgNorm = cv2.normalize(HarrisImg, 0, 255, cv2.NORM_MINMAX);
HarrisImgNormScaled = cv2.convertScaleAbs(HarrisImgNorm);

# Vykresleni krouzku
r, c = HarrisImgNorm.shape
thresh = 5
for j in range(r):
    for i in range(c):
        if(HarrisImgNormScaled[j, i] > thresh):
            cv2.circle(Img, (i, j), 5, (0, 0, 255), 1, 8, 0)

# Zobrazeni vysledku
cv2.namedWindow("Harris Corners - Circle", cv2.WINDOW_NORMAL)
cv2.imshow("Harris Corners - Circle", Img)
cv2.waitKey(0)

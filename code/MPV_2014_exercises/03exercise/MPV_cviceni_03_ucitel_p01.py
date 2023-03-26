# -*- coding: utf-8 -*-
"""
Created on Wed Oct 08 12:59:17 2014

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@version: 1.0.0
"""


# Import modulu.
import cv2 
import numpy as np 
import copy


#------------------------------------------------------------------------------
# Priklad 1: Harris Corner Detector.
#------------------------------------------------------------------------------

# Nacteni barevneho obrazku a prevedeni do odstinu sedi.
Img = cv2.imread("homer.jpg", cv2.IMREAD_COLOR)
ImgGS = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)

# Vykresleni nacteneho obrazku.
cv2.namedWindow("Original Image")
cv2.imshow("Original Image", Img)    
cv2.waitKey(1)

# Vypocet Harris Corner Detectoru.
ImgGS = np.float32(ImgGS)
blockSize = 2;
apertureSize = 3;
k = 0.04;
HarrisImg = cv2.cornerHarris(ImgGS, blockSize, apertureSize, k);

# Vykresleni vysledku.
cv2.namedWindow("Harris Corners")
cv2.imshow("Harris Corners", HarrisImg)    
cv2.waitKey(1)

# Normalizace        
NormHarris = HarrisImg + np.abs(HarrisImg.min())
NormHarris = 255 * (NormHarris / NormHarris.max())

th = 200
_, ImgTh = cv2.threshold(NormHarris, th, 255, cv2.THRESH_BINARY)

# Vykresleni krouzku.
PT = []
r, c = ImgTh.shape
for j in range(r):
    for i in range(c):
        if(ImgTh[j, i] != 0):
            cv2.circle(Img, (i, j), 7, (0, 0, 255), -1, 8, 0)
            PT.append((i, j))

# Vytvoreni hluboke kopie.
Img2 = copy.deepcopy(Img)

# Vykresleni vysledku.
cv2.namedWindow("Harris Corners - Circles")
cv2.imshow("Harris Corners - Circles", Img)
cv2.waitKey(0)

#------------------------------------------------------------------------------
# Priklad 2: Spojeni bodu primkami.
#------------------------------------------------------------------------------

# Spojeni bodu pomoci car.
for i in range(len(PT)):
    if (i + 1) < len(PT):
        cv2.line(Img2, PT[i], PT[i + 1], (0, 255, 0), 3, cv2.CV_AA, 0)
    else:
        cv2.line(Img2, PT[i], PT[0], (0, 255, 0), 3, cv2.CV_AA, 0)

# Vykresleni vysledku.
cv2.namedWindow("Harris Corners - Connected Circles")
cv2.imshow("Harris Corners - Connected Circles", Img2)
cv2.waitKey(0)

cv2.destroyAllWindows()








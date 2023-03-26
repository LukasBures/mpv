# -*- coding: utf-8 -*-
"""
Created on Fri Sep 06 15:51:00 2014

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@version: 2.0.0
"""

# Import OpenCV.
import cv2


# ------------------------------------------------------------------------------
# Nacteni obrazku, zobrazeni a prevedeni do sedotonoveho formatu.
# ------------------------------------------------------------------------------

# Nacteni barevneho obrazku.
# lena.jpg homer.jpg mms.jpg
Img = cv2.imread("./img/homer.jpg", cv2.IMREAD_COLOR)

# Vytvoreni pojmenovaneho okna.
cv2.namedWindow("Original Image")

# Urceni, do jakeho okna se ma dany obrazek vykreslit.
cv2.imshow("Original Image", Img)

# Prevede barevny obrazek do sedotonoveho formatu.
ImgGS = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)

# Nyni se obrazek vykresli na nekonecne dlouhou dobu v [ms], dokud nebude
# stisknuta libovolna klavesa.
cv2.waitKey(0)


# ------------------------------------------------------------------------------
# Priklad 1: Scale-Invariant Feature Transform (SIFT)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Priklad 2: Speeded-Up Robust Features (SURF)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Priklad 3: Oriented FAST and Rotated BRIEF (ORB)
# Features from Accelerated Segment Test (FAST)
# Binary Robust Independent Elementary Features (BRIEF)
# ------------------------------------------------------------------------------

cv2.destroyAllWindows()

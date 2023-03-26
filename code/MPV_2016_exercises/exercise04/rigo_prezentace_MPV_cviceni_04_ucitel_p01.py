# -*- coding: utf-8 -*-
"""
Created on Fri Sep 06 15:51:00 2014

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@version: 3.0.0

Revision Note:
3.0.0 - 18.10.2016 - Updated for OpenCV 3.1.0 version
"""

# Import OpenCV.
import cv2


# ----------------------------------------------------------------------------------------------------------------------
# Nacteni obrazku, zobrazeni a prevedeni do sedotonoveho formatu.

# Nacteni barevneho obrazku.
# lena.jpg homer.jpg mms.jpg
Img = cv2.imread("./img/leto.png", cv2.IMREAD_COLOR)

# Vytvoreni pojmenovaneho okna.
cv2.namedWindow("Original Image")

# Urceni, do jakeho okna se ma dany obrazek vykreslit.
cv2.imshow("Original Image", Img)

# Prevede barevny obrazek do sedotonoveho formatu.
ImgGS = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)

# Nyni se obrazek vykresli na nekonecne dlouhou dobu v [ms], dokud nebude
# stisknuta libovolna klavesa.
cv2.waitKey(0)


# ----------------------------------------------------------------------------------------------------------------------
# Priklad 1: Scale-Invariant Feature Transform (SIFT)

# Vytvoreni SIFT objektu.
# OpenCV 2.7.X
# sift = cv2.SIFT(0, 3, 0.04, 10, 1.6)
# OpenCV 3.1.0
sift = cv2.xfeatures2d.SIFT_create(0, 3, 0.04, 10, 1.6)

# Detekce klicovych bodu.
kpSIFT = sift.detect(ImgGS, None)

# Vykresli vsechny klicove body, jako male cervene krouzky.
ImgSIFT = cv2.drawKeypoints(Img, kpSIFT[:], None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)

# Vytvoreni pojmenovaneho okna.
cv2.namedWindow("SIFT")

# Urceni, do jakeho okna se ma dany obrazek vykreslit.
cv2.imshow("SIFT", ImgSIFT)

# Vypise pocet nalezenych vyznamnych bodu pomoci metody SIFT.
print "-----------------------------------------------------------------------"
print "Pocet nalezenych vyznamnych bodu pomoci detektoru SIFT =", len(kpSIFT)

# Nyni se obrazek vykresli na nekonecne dlouhou dobu v [ms], dokud nebude
# stisknuta libovolna klavesa.
cv2.waitKey(0)


# ----------------------------------------------------------------------------------------------------------------------
# Priklad 2: Speeded-Up Robust Features (SURF)

# Vytvoreni SURF objektu.
# OpenCV 2.7.X
# surf = cv2.SURF(400, 4, 2, True, False)
# OpenCV 3.1.0
surf = cv2.xfeatures2d.SURF_create(400, 4, 2, True, False)

# Detekce klicovych bodu.
kpSURF = surf.detect(Img, None)

# Vykresli prvnich vsech klicovych bodu, jako cervene kruhy
# s velikosti a orientaci.
ImgSURF = cv2.drawKeypoints(Img, kpSURF[:], None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Vytvoreni pojmenovaneho okna.
cv2.namedWindow("SURF")

# Urceni, do jakeho okna se ma dany obrazek vykreslit.
cv2.imshow("SURF", ImgSURF)

# Vypise pocet nalezenych vyznamnych bodu pomoci metody SURF.
print "-----------------------------------------------------------------------"
print "Pocet nalezenych vyznamnych bodu pomoci detektoru SURF =", len(kpSURF)

# Nyni se obrazek vykresli na nekonecne dlouhou dobu v [ms], dokud nebude
# stisknuta libovolna klavesa.
cv2.waitKey(0)


# ----------------------------------------------------------------------------------------------------------------------
# Priklad 3: Oriented FAST and Rotated BRIEF (ORB)
# Features from Accelerated Segment Test (FAST)
# Binary Robust Independent Elementary Features (BRIEF)

# Pocet hledanych vyznamnych bodu.
nORB = 100

# Vytvoreni ORB objektu.
# OpenCV 2.7.X
# orb = cv2.ORB(nORB, 1.2, 8, 31, 0, 2, cv2.ORB_HARRIS_SCORE, 31)
# OpenCV 3.1.0
orb = cv2.ORB_create(nORB, 1.2, 8, 31, 0, 2, cv2.ORB_HARRIS_SCORE, 31)

# Detekce klicovych bodu.
kpORB = orb.detect(Img, None)

# Vykresli prvnich vsech klicovych bodu, jako cervene kruhy
# s velikosti a orientaci.
ImgORB = cv2.drawKeypoints(Img, kpORB[:], None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Vytvoreni pojmenovaneho okna.
cv2.namedWindow("ORB")

# Urceni, do jakeho okna se ma dany obrazek vykreslit.
cv2.imshow("ORB", ImgORB)

# Vypise pocet nalezenych vyznamnych bodu pomoci metody ORB.
print "-----------------------------------------------------------------------"
print "Pocet nalezenych vyznamnych bodu pomoci detektoru ORB =", len(kpORB)

# Nyni se obrazek vykresli na nekonecne dlouhou dobu v [ms], dokud nebude
# stisknuta libovolna klavesa.
cv2.waitKey(0)

cv2.destroyAllWindows()

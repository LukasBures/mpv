# -*- coding: utf-8 -*-
"""
Created on Thu Sep 04 16:21:27 2014

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits: Ing. Marek Hruz, Ph.D.
@version: 3.0.0

Revision Note:
3.0.0 - 23.9.2016 - Updated for OpenCV 3.1.0 version
"""


# Import OpenCV modulu.
import cv2 

# Import modulu pro pocitani s Pythonem.
import numpy as np 

# Import modulu pro vykreslovani grafu.
from matplotlib import pyplot as plt 


# ----------------------------------------------------------------------------------------------------------------------
# Priklad 1: Nacteni a zobrazeni barevneho obrazku.
# ----------------------------------------------------------------------------------------------------------------------

# Nacteni barevneho obrazku.
mmsColor = cv2.imread("./img/mms.jpg", cv2.IMREAD_COLOR)

# Vytvoreni pojmenovaneho okna.
cv2.namedWindow("Original image")

# Urceni, do jakeho okna se ma dany obrazek vykreslit.
cv2.imshow("Original image", mmsColor)

# Nyni se obrazek vykresli na nekonecne dlouhou dobu v [ms], dokud nebude stisknuta libovolna klavesa.
cv2.waitKey(0)


# ----------------------------------------------------------------------------------------------------------------------
# Priklad 2: Nacteni barevneho obrazku a nasledna konverze do sedotonu.
# ----------------------------------------------------------------------------------------------------------------------

# Prevede barevny obrazek do sedotonoveho formatu.
mmsGS = cv2.cvtColor(mmsColor, cv2.COLOR_RGB2GRAY)

# Vytvoreni pojmenovaneho okna.
cv2.namedWindow("Grayscale image")

# Urceni, do jakeho okna se ma dany obrazek vykreslit.
cv2.imshow("Grayscale image", mmsGS)

# Nyni se obrazek vykresli na nekonecne dlouhou dobu v [ms], dokud nebude stisknuta libovolna klavesa.
cv2.waitKey(0)

# Znici vsechna vytvorena okna.
cv2.destroyAllWindows()


# ----------------------------------------------------------------------------------------------------------------------
# Priklad 3: Nacteni a vykresleni sedotonoveho obrazku, vypocet histogramu a distribucni funkce.
# ----------------------------------------------------------------------------------------------------------------------

# Nacteni sedotonoveho obrazku.
ImgGS = cv2.imread("./img/tsukuba_r.png", cv2.IMREAD_GRAYSCALE)
# ImgGS = cv2.imread("Unequalized_Hawkes_Bay_NZ.jpg", cv2.IMREAD_GRAYSCALE)
# ImgGS = cv2.imread("rice.jpg", cv2.IMREAD_GRAYSCALE)

# Vypocita histogram z obrazku.
hist, bins = np.histogram(ImgGS.flatten(), 256, [0, 256])

# Vypocita kumulativni soucet histogramu (distribucni funkci). Cumulative Distribution Function (cdf).
cdf = hist.cumsum()

# Normalizace pro vykresleni do histogramu.
cdf_normalized = cdf * hist.max() / cdf.max()

# Vytvori novou figuru, stejne jako v MATLABu.
h = plt.figure("Histogram")

# Vykresleni distribucni funkce histogramu modrou barvou.
plt.plot(cdf_normalized, color='b')

# Vykresleni sloupcu histogramu cervenou barvou.
plt.hist(ImgGS.flatten(), 256, [0, 256], color='r')

# Nastaveni limitu X-ove osy.
plt.xlim([0, 256])

# Vykresleni legendy.
plt.legend(('Cumulative Distribution Function', 'Histogram'), loc='upper left')

# Vykresleni grafu.
plt.show()

# Vytvoreni pojmenovaneho okna.
cv2.namedWindow("Unequalized Image")

# Urceni, do jakeho okna se ma dany obrazek vykreslit.
cv2.imshow("Unequalized Image", ImgGS)

# Nyni se obrazek vykresli na nekonecne dlouhou dobu v [ms], dokud nebude stisknuta libovolna klavesa.
cv2.waitKey(0)

# Zavre vsechna vytvorena OpenCV okna.
cv2.destroyAllWindows()


# ----------------------------------------------------------------------------------------------------------------------
# Priklad 4: Globalni equalizace histogramu
# ----------------------------------------------------------------------------------------------------------------------

# Spocita globalni equalizaci histogramu.
ImgGSEQ = cv2.equalizeHist(ImgGS)

# Vytvori histogram z obrazku.
histEQ, binsEQ = np.histogram(ImgGSEQ.flatten(), 256, [0, 256])

# Vypocita kumulativni soucet histogramu (distribucni funkci).
# Cumulative Distribution Function (cdf)
cdfEQ = histEQ.cumsum()

# Normalizace.
cdf_normalized_EQ = cdfEQ * histEQ.max() / cdfEQ.max()

# Vytvori novou figuru, stejne jako v MATLABu.
plt.figure("Globally Equalized Histogram")

# Vykresleni distribucni funkce histogramu modrou barvou.
plt.plot(cdf_normalized_EQ, color='b')

# Vykresleni sloupcu equalizovaneho histogramu cervenou barvou.
plt.hist(ImgGSEQ.flatten(), 256, [0, 256], color='r')

# Nastaveni limitu X-ove osy.
plt.xlim([0, 256])

# Vykresleni legendy.
plt.legend(('Cumulative Distribution Function', 'Globally Equalized Histogram'), loc='upper left')

# Vykresleni grafu.
plt.show()

# Spoji dva obrazky vedle sebe a vytvori z nich jeden
SideBySide = np.hstack((ImgGS, ImgGSEQ))

# Vytvoreni pojmenovaneho okna.
cv2.namedWindow("Original Image VS. Globally Equalized Image")

# Urceni, do jakeho okna se ma dany obrazek vykreslit.
cv2.imshow("Original Image VS. Globally Equalized Image", SideBySide)

# Nyni se obrazek vykresli na nekonecne dlouhou dobu v [ms], dokud nebude stisknuta libovolna klavesa.
cv2.waitKey(0)

# Zavre vsechna vytvorena OpenCV okna.
cv2.destroyAllWindows()


# ------------------------------------------------------------------------------
# Priklad 5: Contrast Limited Adaptive Histogram Equalization (CLAHE)
# ------------------------------------------------------------------------------

# Vytvori CLAHE objekt s volitelnymi parametry.
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Aplikuje CLAHE s nastavenymi parametry na vstupni obrazek.
ImgCLAHE = CLAHE.apply(ImgGS)

# Vytvori histogram z obrazku.
histCLAHE, binsCLAHE = np.histogram(ImgCLAHE.flatten(), 256, [0, 256])

# Vypocita kumulativni soucet histogramu (distribucni funkci). Cumulative Distribution Function (cdf)
cdfCLAHE = histCLAHE.cumsum()

# Normalizace.
cdf_normalized_CLAHE = cdfCLAHE * histCLAHE.max() / cdfCLAHE.max()

# Vytvori novou figuru, stejne jako v MATLABu.
plt.figure("Contrast Limited Adaptive Histogram Equalization (CLAHE)")

# Vykresleni distribucni funkce histogramu modrou barvou.
plt.plot(cdf_normalized_CLAHE, color='b')

# Vykresleni sloupcu equalizovaneho histogramu cervenou barvou.
plt.hist(ImgCLAHE.flatten(), 256, [0, 256], color='r')

# Nastaveni limitu X-ove osy.
plt.xlim([0, 256])

# Vykresleni legendy.
plt.legend(('Cumulative Distribution Function', 'CLAHE'), loc='upper left')

# Vykresleni grafu.
plt.show()

# Spoji dva obrazky vedle sebe a vytvori z nich jeden
SideBySide = np.hstack((SideBySide, ImgCLAHE))

# Vytvoreni pojmenovaneho okna.
cv2.namedWindow("Original Image VS. Globally Equalized Image VS. CLAHE")

# Urceni, do jakeho okna se ma dany obrazek vykreslit.
cv2.imshow("Original Image VS. Globally Equalized Image VS. CLAHE", SideBySide)

# Nyni se obrazek vykresli na nekonecne dlouhou dobu v [ms], dokud nebude stisknuta libovolna klavesa.
cv2.waitKey(0)


# ----------------------------------------------------------------------------------------------------------------------
# Zavreni vsech oken.
# ----------------------------------------------------------------------------------------------------------------------

# Zavre vsechna vytvorena OpenCV okna.
cv2.destroyAllWindows()

# Zavre figuru s nazvem "Histogram".
plt.close("Histogram")

# Zavre figuru s nazvem "Globally Equalized Histogram".
plt.close("Globally Equalized Histogram")

# Zavre figuru s nazvem "Contrast Limited Adaptive Histogram Equalization (CLAHE)".
plt.close("Contrast Limited Adaptive Histogram Equalization (CLAHE)")


# ----------------------------------------------------------------------------------------------------------------------
# Otsuova metoda automatickeho prahovani.
# ----------------------------------------------------------------------------------------------------------------------

# Nacteni sedotonoveho obrazku.
ImgGS_rice = cv2.imread("./img/rice.jpg", cv2.IMREAD_GRAYSCALE)

# Vytvoreni pojmenovaneho okna.
cv2.namedWindow("Rice")

# Urceni, do jakeho okna se ma dany obrazek vykreslit.
cv2.imshow("Rice", ImgGS_rice)

# Vytvori CLAHE objekt s volitelnymi parametry.
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Aplikuje CLAHE objekt na obrazek.
ImgCLAHE_rice = CLAHE.apply(ImgGS_rice)

# Pouzije Otsuovu metodu prahovani.
retval, ImgThOtsu = cv2.threshold(ImgCLAHE_rice, 0, 255, cv2.THRESH_OTSU)

# Vytvoreni pojmenovaneho okna.
cv2.namedWindow("Otsu's Threshold")

# Urceni, do jakeho okna se ma dany obrazek vykreslit.
cv2.imshow("Otsu's Threshold", ImgThOtsu)

# Nyni se obrazek vykresli na nekonecne dlouhou dobu v [ms], dokud nebude stisknuta libovolna klavesa.
cv2.waitKey(0)

# Vytvori strukturni element o velikosti 3x3 s pocatkem uprostred.
strElement = np.ones((3, 3), np.uint8)

# Provede 1 iteraci morfologicke operace otevreni s vyse definovanym strukturnim elementem.
ImgThOtsuOpen = cv2.morphologyEx(ImgThOtsu, cv2.MORPH_OPEN, strElement, 1)

# Nalezne kontury.
_, contours, _  = cv2.findContours(ImgThOtsuOpen, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Pokud je pocet kontur vetsi nez nula, tak se vykresli a vypise pocet zrn ryze na obrazku.
if len(contours) > 0:
    # Zjisti velikost obrazku.
    height, width = ImgThOtsuOpen.shape

    # Vytvori novy barevny obrazek o dane velikosti.
    contoursDraw = np.zeros((height, width, 3), np.uint8)
    
    # Vykresli nalezene kontury do vytvoreneho obrazku, 3. parametr -1 znamena, ze budou vykresleny vsechny kontury,
    # cervenou barvou (BGR) a 5. parametr znamena, ze kontury budou vykresleny vyplnene.
    cv2.drawContours(contoursDraw, contours, -1, (0, 0, 255), -1)
    
    # Vypise pocet nalezenych kontur = pocet zrn ryze.
    print "Pocet zrnek ryze =", len(contours)    
    
    # Vytvoreni pojmenovaneho okna.
    cv2.namedWindow("Draw Contours")
    
    # Urceni, do jakeho okna se ma dany obrazek vykreslit.
    cv2.imshow("Draw Contours", contoursDraw)
    
    # Nyni se obrazek vykresli na nekonecne dlouhou dobu v [ms], dokud nebude stisknuta libovolna klavesa.
    cv2.waitKey(0)
    
else:
    # Vypise pocet nalezenych kontur = pocet zrn ryze.
    print "Pocet zrnek ryze =", len(contours)    
    
# Zavre vsechna vytvorena OpenCV okna.
cv2.destroyAllWindows()

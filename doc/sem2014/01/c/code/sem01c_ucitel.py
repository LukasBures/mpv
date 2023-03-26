# -*- coding: utf-8 -*-
"""
Created on Fri Sep 05 12:33:44 2014

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits: Ing. Marek Hruz, Ph.D.
@version: 1.0.0
"""


import cv2
import numpy as np
import warnings
from scipy import signal

#------------------------------------------------------------------------------
# Vypocte odezvy ve vodorovnem a svislem smeru na Sobeluv operator (derivace).
# Vstupni sedotonovy obrazek a velikost Sobelova filtru (3, 5 nebo 7).
#------------------------------------------------------------------------------
def Sobel(ImgGS, kSize):
    
    # Velikost kernelu musi byt licha a >= 3.
    if (kSize % 2 == 0) | (kSize < 3):
        warnings.warn("Spatna velikost masky! Zvolte liche cislo >= 3.")   
        kSize = 3
        print("Byla zvolena Sobelova maska o velikosti 3x3.")

    # Tvorba Sobelova kernelu.    
    if kSize == 3:
        xKernel = np.float64(np.multiply([[1], [2], [1]], [1, 0, -1]))
        yKernel = np.float64(np.multiply([[1], [0], [-1]], [1, 2, 1]))
    else:
        xKernel = np.float64(np.multiply([[1], [2], [1]], [1, 0, -1]))
        yKernel = np.float64(np.multiply([[1], [0], [-1]], [1, 2, 1]))
        smooth = np.float64(np.multiply([[1.], [2.], [1.]], [1., 2., 1.]) / 8.)
        nIter = (kSize - 3) / 2

        for i in range(nIter):
            xKernel = signal.convolve2d(xKernel, smooth)
            yKernel = signal.convolve2d(yKernel, smooth)
   
    # Provede 2D konvoluci s vytvorenym maskou.
    Ix = cv2.filter2D(ImgGS, cv2.CV_64F, xKernel)
    Iy = cv2.filter2D(ImgGS, cv2.CV_64F, yKernel)
    
    # Hrany z jednoho smeru maji kladnou odezvu a hrany z druheho smeru maji
    # zapornou odezvu na Sobeluv filtr, proto je nutne pouzit absolutni hodnotu
    # abychom dostali vsechny hrany.
#    Ix = np.absolute(Ix)
#    Iy = np.absolute(Iy)
    
    # Vrati odezvy na jednotlive filtry.
    return Ix, Iy
    
    
#------------------------------------------------------------------------------
# Vstupni sedotonovy obrazek, velikost okynka, velikost pro Sobel operator.
#------------------------------------------------------------------------------
def cornerHarris(ImgGS, blockSize, kSize, threshold):
    
    # Zavola funkci, ktera vypocita odezvy na Sobelovy filtry.
    Ix, Iy = Sobel(ImgGS, kSize)    
    
#    cv2.imshow("Ix", np.uint8(Ix))
#    cv2.imshow("Iy", np.uint8(Iy))
    
    # Vypocita velikost.    
    I = np.sqrt(np.power(Ix, 2) + np.power(Iy, 2))
       
    # Urceni, do jakeho okna se ma dany obrazek vykreslit.
    cv2.imshow("I", np.uint8(I))       

    # Zjisti, jestli je velikost okenka liche cislo.    
    if blockSize % 2 == 0:
        warnings.warn("Velikost okenka musi byt licha!")
        blockSize = 3
        print("Nastavuji velikost okna na velikosti 3x3.")


    # Vytvoreni Gaussovskeho kernelu.
    kernelX = cv2.getGaussianKernel(blockSize, 0)
    kernelY = cv2.getGaussianKernel(blockSize, 0)
    gaussKernel = kernelX * kernelY.T

    
 #   print gaussKernel

    # Konvuluce s gaussovskym kernelem.
    Mxx = cv2.filter2D(np.power(Ix, 2), cv2.CV_64F, gaussKernel)
    Mxy = cv2.filter2D(np.multiply(Ix, Iy), cv2.CV_64F, gaussKernel)
    Myy = cv2.filter2D(np.power(Iy, 2), cv2.CV_64F, gaussKernel)
    
    Mdet = Mxx * Myy - np.power(Mxy, 2)
    Mtr = Mxx + Myy
    k = 0.04
    R = Mdet - k * np.power(Mtr, 2)
    
    #R = np.uint8(R)    
#      
#    cv2.imshow("RRRRRRRRRRRRR", R) 
#    cv2.waitKey(1)

    print "---My--------------------------------------------------------------"
    print np.min(R), np.max(R)
    NormHarris = R + np.abs(R.min())
    NormHarris = 255 * (NormHarris / NormHarris.max())
    print np.min(NormHarris), np.max(NormHarris)
    _, ImgTh = cv2.threshold(np.uint8(NormHarris), threshold, 255, cv2.THRESH_BINARY)
    print "-------------------------------------------------------------------"    
    cv2.namedWindow("My", 0)
    cv2.imshow("My", ImgTh) 
    cv2.waitKey(1)

    return ImgTh

    
#-----------------------------------------------------------------------------
# Neni dovoleno menit metodu main.
#-----------------------------------------------------------------------------
if __name__ == '__main__':
    ImgGS = cv2.imread("homer.jpg", cv2.IMREAD_GRAYSCALE)
    blockSize = 3
    kSize = 3
    threshold = 100
    
    outMyImg = cornerHarris(ImgGS, blockSize, kSize, threshold)
    out = cv2.cornerHarris(ImgGS, blockSize, kSize, 0.04)
    
    print "---OpenCV----------------------------------------------------------"
    print np.min(out), np.max(out)
    NormHarris = out + np.abs(out.min())
    NormHarris = 255 * (NormHarris / NormHarris.max()) 
    print np.min(NormHarris), np.max(NormHarris)
    print "-------------------------------------------------------------------"
    
    _, ImgTh = cv2.threshold(np.uint8(NormHarris), threshold, 255, cv2.THRESH_BINARY)
    cv2.namedWindow("OpenCV", 0)
    cv2.imshow("OpenCV", ImgTh) 
    cv2.waitKey(1)
        
    Rozdilovy = np.abs(outMyImg - out)
    cv2.namedWindow("Rozdilovy", 0)
    cv2.imshow("Rozdilovy", Rozdilovy) 
    cv2.waitKey(0)
            
    
    
    # Zavre vsechna vytvorena OpenCV okna.
    cv2.destroyAllWindows()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
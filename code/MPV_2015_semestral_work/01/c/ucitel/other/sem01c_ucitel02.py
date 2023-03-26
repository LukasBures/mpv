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
  
    
#------------------------------------------------------------------------------
# Harris Corner Detector
#------------------------------------------------------------------------------
def cornerHarris(Img, blockSize, kSize, k, threshold):
    
    # Prevod obrazku do sedotonu
    ImgGS = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)

    # Harris Corner Detector    
    ImgHarris = cv2.cornerHarris(ImgGS, blockSize, kSize, k)

    # Normalizace        
    NormHarris = ImgHarris + np.abs(ImgHarris.min())
    NormHarris = 255 * (NormHarris / NormHarris.max())
      
    _, ImgTh = cv2.threshold(NormHarris, threshold, 255, cv2.THRESH_BINARY)
 
    return ImgTh
    
#-----------------------------------------------------------------------------
# Neni dovoleno menit metodu main.
#-----------------------------------------------------------------------------
if __name__ == '__main__':
    """
    
    """
    
    # Nacteni barevneho obrazku.
    Img = cv2.imread("homer.jpg", cv2.IMREAD_COLOR)

    # Urceni, do jakeho okna se ma dany obrazek vykreslit.
    cv2.namedWindow("Original Image")
    cv2.imshow("Original Image", Img)
    cv2.waitKey(1)
    
    # Parametry pro Harris Corner Detector
    blockSize = 3
    kSize = 3
    threshold = 50 # MUSI SE NACIST ZE SERVRU - hodnoty 0-255, ale tak, aby tam neco bylo videt
    k = 0.04
    ImgTh = cornerHarris(Img, blockSize, kSize, k, threshold)

    cv2.namedWindow("ImgTh")
    cv2.imshow("ImgTh", ImgTh)
    cv2.waitKey(0)


    # Zavre vsechna vytvorena OpenCV okna.
    cv2.destroyAllWindows()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
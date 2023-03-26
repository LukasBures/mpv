# -*- coding: utf-8 -*-
"""
Created on Wed Oct 08 12:59:17 2014

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@version: 1.0.0
"""


# Import modulu.
import cv2 

#-----------------------------------------------------------------------------
# Priklad 1: Canny edge detector.
#-----------------------------------------------------------------------------
def CannyThreshold(srcGS):
    kernelSize = 3
    LowThreshold = 50
    HighThreshold = 150
    
    # Redukce sumu rozmazanim.
    edgesImg = cv2.blur(srcGS, (3, 3));

    # Canny edges detector.
    Edges = cv2.Canny(edgesImg, LowThreshold, HighThreshold, kernelSize);

    return Edges

if __name__ == '__main__':
    # Nacteni barevneho obrazku a prevedeni do odstinu sedi
    Img = cv2.imread("homer.jpg", cv2.IMREAD_COLOR)
    ImgGS = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)
    #_, ImgGS = cv2.threshold(ImgGS, 0, 255, cv2.THRESH_OTSU)
    
    # Vykresleni
    cv2.namedWindow("Original Image")#, cv2.WINDOW_NORMAL
    cv2.imshow("Original Image", Img)    
    cv2.waitKey(1)
    
    EdgesImg = CannyThreshold(ImgGS)    
    
    cv2.namedWindow("Canny Edges")#, cv2.WINDOW_NORMAL
    cv2.imshow("Canny Edges", EdgesImg)    
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()





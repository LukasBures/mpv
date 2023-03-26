# -*- coding: utf-8 -*-
"""
Created on Wed Oct 08 12:59:17 2014

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@version: 2.0.0
"""

# Import modulu.
import cv2 

# -----------------------------------------------------------------------------
# Priklad 1: Canny edge detector.
# -----------------------------------------------------------------------------


def canny_threshold(src_gs):
    """
    Funkce pro vypocet

    :param src_gs: Vstupni sedotonovy obrazek.
    :type src_gs: ndarray uint8

    :return edges: Vystupni binarni obrazek nalezenych hran.
    :rtype edges: ndarray uint8
    """
    kernel_size = 3
    low_threshold = 50
    high_threshold = 150
    
    # Redukce sumu rozmazanim.
    edges_img = cv2.blur(src_gs, (3, 3))

    # Canny edges detector.
    edges = cv2.Canny(edges_img, low_threshold, high_threshold, kernel_size)

    return edges

if __name__ == '__main__':
    # Nacteni barevneho obrazku a prevedeni do odstinu sedi
    Img = cv2.imread("./img/homer.jpg", cv2.IMREAD_COLOR)
    ImgGS = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)
    # ImgGS = cv2.threshold(ImgGS, 0, 255, cv2.THRESH_OTSU)
    
    # Vykresleni
    cv2.namedWindow("Original Image", cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("Original Image", 100, 100)
    cv2.imshow("Original Image", Img)    
    cv2.waitKey(1)
    
    EdgesImg = canny_threshold(ImgGS)
    
    cv2.namedWindow("Canny Edges", cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("Canny Edges", 100, 100)
    cv2.imshow("Canny Edges", EdgesImg)    
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()

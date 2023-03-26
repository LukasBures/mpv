# -*- coding: utf-8 -*-
"""
Created on 13:40 6.10.15 

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits: 
@version: 2.0.0
"""

import warnings
import numpy as np
import cv2 # konvoluce


def konvoluce():
    a = 1
    return a


def sobel(gsimg, ksize):
    """
    Vypocte odezvy ve vodorovnem a svislem smeru na Sobeluv operator (derivace).
    Vstupni sedotonovy obrazek a velikost Sobelova filtru (3, 5 nebo 7).

    :param gsimg:
    :param ksize:
    :return:
    """

    # Velikost kernelu musi byt licha a >= 3.
    if (ksize % 2 == 0) | (ksize < 3):
        warnings.warn("Spatna velikost masky! Zvolte liche cislo >= 3.")
        ksize = 3
        print("Byla zvolena Sobelova maska o velikosti 3x3.")

    # Tvorba Sobelova kernelu.
    if ksize == 3:
        x_kernel = np.float32(np.multiply([[1], [2], [1]], [-1, 0, 1]))
        y_kernel = np.float32(np.multiply([[-1], [0], [1]], [1, 2, 1]))
    else:
        warnings.warn("ksize > 3 is not implemented, using ksize = 3")
        x_kernel = np.float32(np.multiply([[1], [2], [1]], [-1, 0, 1]))
        y_kernel = np.float32(np.multiply([[-1], [0], [1]], [1, 2, 1]))
    # else:
    #     x_kernel = np.float32(np.multiply([[1], [2], [1]], [-1, 0, 1]))
    #     y_kernel = np.float32(np.multiply([[-1], [0], [1]], [1, 2, 1]))
    #     smooth = np.float32(np.multiply([[1.], [2.], [1.]], [1., 2., 1.]) / 8.)
    #     n_iter = (ksize - 3) / 2
    #
    #     for i in range(n_iter):
    #         x_kernel = signal.convolve2d(x_kernel, smooth)
    #         y_kernel = signal.convolve2d(y_kernel, smooth)

    # Provede 2D konvoluci s vytvorenym maskou.
    ix = cv2.filter2D(gsimg, cv2.CV_32F, x_kernel, anchor=(-1, -1), borderType=cv2.BORDER_DEFAULT)
    iy = cv2.filter2D(gsimg, cv2.CV_32F, y_kernel, anchor=(-1, -1), borderType=cv2.BORDER_DEFAULT)

    # Vrati odezvy na jednotlive filtry.
    return ix, iy


def harris(gsimg, block_size, ksize, k):
    """
    Metoda aplikuje Harrisuv detektor vcholu na vstupni obrazek a vrati vysledky ve forme obrazku.

    :param gsimg: Vstupni sedotonovy obrazek.
    :type gsimg: 2D ndarray of uint8

    :param block_size: Velikost kernelu pro rozmazani. Liche cislo >= 3.
    :type block_size: int

    :param ksize: Velikost kernelu Sobelovo operatoru. Liche cislo >= 3.
    :type ksize: int

    :param k: Harrisuv volny parametr v rovnicich.
    :type k: float

    :return corners: Vystupni obrazek s detekovanymi vrcholy (normalizace na hodnoty 0-255)
    :rtype corners: 2D ndarray of uint8
    """

    # Zavola funkci, ktera vypocita odezvy na Sobelovy filtry.
    ix, iy = sobel(gsimg, ksize)
    # sobelx = cv2.Sobel(gsimg, cv2.CV_32F, 1, 0, ksize=ksize, borderType=cv2.BORDER_DEFAULT)
    # sobely = cv2.Sobel(gsimg, cv2.CV_32F, 0, 1, ksize=ksize, borderType=cv2.BORDER_DEFAULT)
    #
    # if np.array_equal(ix, sobelx):
    #     print "OK Sobel x"
    # if np.array_equal(iy, sobely):
    #     print "OK Sobel y"

    # Zjisti, jestli je velikost okenka liche cislo.
    if (block_size % 2 == 0) | (block_size < 3):
        warnings.warn("Velikost okenka musi byt licha, zvolte cislo >= 3.")
        block_size = 3
        print("Nastavuji velikost okna na velikosti 3x3.")

    kernel = np.multiply(np.ones((block_size, block_size), dtype=np.float32),
                         np.float32(1.0 / (block_size * block_size)))

    # # Konvuluce s box filtrem.
    mxx = cv2.filter2D(np.power(ix, 2), cv2.CV_32F, kernel, anchor=(-1, -1), borderType=cv2.BORDER_DEFAULT)
    mxy = cv2.filter2D(np.multiply(ix, iy), cv2.CV_32F, kernel, anchor=(-1, -1), borderType=cv2.BORDER_DEFAULT)
    myy = cv2.filter2D(np.power(iy, 2), cv2.CV_32F, kernel, anchor=(-1, -1), borderType=cv2.BORDER_DEFAULT)

    # Determinant matice M
    # mdet = np.subtract(np.multiply(mxx, myy), np.power(mxy, 2))
    mdet = (mxx * myy) - (mxy * mxy)

    # Trace matice M
    # mtr = np.add(mxx, myy)
    mtr = mxx + myy

    # Vysledny obrazek se zvyraznenymi vrcholy
    # corners = np.subtract(mdet, np.multiply(np.float32(k), np.power(mtr, 2)))
    corners = mdet - (k * (mtr * mtr))

    return corners

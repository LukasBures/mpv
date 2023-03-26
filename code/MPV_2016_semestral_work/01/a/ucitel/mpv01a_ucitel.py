# -*- coding: utf-8 -*-
"""
Created on Wed Oct 01 17:13:56 2014

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@version: 2.0.0
"""


def otsu(gsimg):
    """
    Vypocet optimalniho prahu pomoci Otsuovo metody.

    :param gsimg: Vstupni sedotonovy obrazek.
    :type gsimg: 2D ndarray of uint8

    :return threshold: Vystupni hodnota optimalniho prahu.
    :rtype threshold: int
    """

    # Pocet pixelu
    total = gsimg.shape[0] * gsimg.shape[1]

    # Vypocet histogramu
    hist = [0] * 256
    for i in range(gsimg.shape[0]):
        for j in range(gsimg.shape[1]):
            hist[gsimg[i][j]] += 1

    s = 0  # sum
    for j in range(256):
        s += j * hist[j]
        
    sb = 0.0  # 2nd sum
    wb = 0.0
    ma = 0.0  # max
    th1 = 0.0
    th2 = 0.0
    
    for j in range(256):
        wb += hist[j]
        
        if wb == 0:
            continue

        wf = total - wb
        
        if wf == 0:
            break
        
        sb += j * hist[j]
        mb = sb / wb
        mf = (s - sb) / wf
        between = wb * wf * (mb - mf) * (mb - mf)
        
        if between >= ma:
            th1 = j
            if between > ma:
                th2 = j
            ma = between

    return int((th1 + th2) / 2.0)

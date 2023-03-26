# -*- coding: utf-8 -*-
"""
Created on Wed Oct 01 17:13:56 2014

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@version: 2.0.0
"""
#import numpy
global hist, str


def aa(k):
    global hist, str
    w0 = sum(hist[:k])
    str0 = 0
    for ind , x in enumerate(hist[:k]):
        str0 += ind * x
    try:
        var = pow((str * w0 - str0),2)/(w0 * (1 - w0))
    except ZeroDivisionError:
        var = 0
    return var


def otsu(gsimg):
    """
    Vypocet optimalniho prahu pomoci Otsuovo metody.

    :param gsimg: Vstupni sedotonovy obrazek.
    :type gsimg: 2D ndarray of uint8

    :return threshold: Vystupni hodnota optimalniho prahu.
    :rtype threshold: int
    """

    # Vase implementace
    global hist, str
    hist = [0] * 256
    pix = 0
    str = 0
    for i in gsimg:
        for j in i:
            hist[j] += 1
            pix += 1
    for ind , x in enumerate(hist):
        hist[ind] = x/float(pix)
        str += x * hist[ind]
    k = 0
    m = 0
    for i in range(1,255):
        kp = aa(i)
        if kp > m:
            print kp
            m = kp
            k = i
    threshold = k
    return threshold


#a = numpy.random.random_integers(0 , 255, [100, 100])
#otsu(a)

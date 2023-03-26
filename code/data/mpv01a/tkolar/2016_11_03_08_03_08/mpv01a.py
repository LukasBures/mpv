# -*- coding: utf-8 -*-
"""
Created on Wed Oct 01 17:13:56 2014

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@version: 2.0.0
"""



global hist, str


def ots(k):
    global hist, str
    w0 = sum(hist[:k])  #pravdepodobnost prvni tridy
    str0 = 0
    for ind , x in enumerate(hist[:k]):
        str0 += ind * x # vypočet stredni hodnoty prvni tridy
    try:
        var = pow((str * w0 - str0), 2)/(w0 * (1 - w0)) # vypocet rozptylu
    except ZeroDivisionError: # osetreni jasu , ktery se v obrazu nevyskytuje
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
    hist = [0] * 256 #tvorba seznamu pro histogram
    pix = 0
    str = 0
    for i in gsimg:
        for j in i: # procházení přes jednotlivé obrazové body
            hist[j] += 1 # přidání jedničky odpovídajícícmu jasu
            pix += 1 # počítání bodu celeho obrazu
    for ind, x in enumerate(hist): # porcházení přes jasy histogramu
        hist[ind] = x/float(pix)   # normalizace histogramu
        str += hist[ind] * ind     # výpočet střední hodnoty histogramu
    k = 0
    m = 0
    for i in range(1,255): # počítání otsuovy metody přes všechny jasy
        kp = ots(i)
        if kp > m: # maximalizace
            m = kp
            k = i
    threshold = k - 1
    return threshold




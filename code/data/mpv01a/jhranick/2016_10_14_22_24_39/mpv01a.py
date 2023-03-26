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
    # Vytvoreni histogramu obrazku pri 256 urovnich sedi
    hist = [0]*256
    for row in gsimg:
        for intensity in row:
            hist[intensity] += 1

    numPixels = float(len(gsimg)**2)                        # pocet pixlu
    numLevels = 256                                         # pocet urovni sedi
    probs = [float(x)/numPixels for x in hist]              # pravdepodobnosti pro jednotlive prvky histogramu
    print('[CHECK] Probability = %f' % sum(probs))

    # Celkova stredni hodnota
    meanT = 0.0
    for i, val in enumerate(hist):
        meanT += float(i*val)
    meanT /= numPixels

    print('[DEBUG] Global mean = %f' % meanT)

    # Otsu's method implementation
    omega = 0.0
    mean = 0.0
    threshold = 0
    sigmaMax = 0
    for k in range(0, numLevels - 1):
        # pi = hist[k]/numPixels
        pi = probs[k]
        omega += pi
        omega1 = 1 - omega

        if omega in (0.0, 1.0):
            continue

        # Vypocet strednich hodnot
        mean += float(k*pi)
        mu0 = mean/omega
        mu1 = (meanT - mean)/(1 - omega)

        # Variance mezi tridami
        sigmaB = omega*omega1*(mu1-mu0)**2

        # Kriterium pro nalezeni optimalnihoo thresholdu
        if sigmaB > sigmaMax:
            sigmaMax = sigmaB   # aktualizace nalezeneho maxima
            threshold = k       # nalezeny threshold
    print('[DEBUG] Threshold found = %d' % threshold)

    return threshold

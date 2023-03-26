from __future__ import division
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

    results = {}
    histogram = {}

    pixelsY = len(gsimg)
    pixelsX = len(gsimg[0])
    pixels = pixelsX * pixelsY

    for tmp in range(0, 256, 1):
        histogram[tmp] = 0

    for i, val in enumerate(gsimg):
        for j, val2 in enumerate(val):
            histogram[val2] += 1

    print histogram

    for k in range(0, 256):
        # k = 3 # smazat a nechat for loop

        # jednotlive vahy histogramu
        wBeg = 0.0
        for i in range(0, k):
            wBeg += histogram[i]
        wBeg /= pixels
        wEnd = 1 - wBeg

        print "k: " + str(k)
        print "wBeg: " + str(wBeg)
        print "wEnd: " + str(wEnd)

        # vypocet str. hodnot
        if wBeg != 0 and wEnd != 0:
            u0 = 0.0
            for i in range(0, k):
                u0 += i * histogram[i] / wBeg

            u1 = 0.0
            for i in range(k, len(histogram)):
                u1 += i * histogram[i] / wEnd

            #print u0
            #print u1

            uT = 0.0
            for i in range(0, len(histogram)):
                uT += i * histogram[i]
            #print uT
            #print wBeg * u0 + wEnd * u1

            # count rozptyly

            sig0 = 0.0
            for i in range(0, k):
                sig0 += (pow(i - u0, 2) * (histogram[i] / pixels)) / wBeg

            sig1 = 0.0
            for i in range(k, len(histogram)):
                sig1 += (pow(i - u1, 2) * (histogram[i] / pixels)) / wEnd

            #print sig0
            #print sig1

            sigB = wBeg * wEnd * pow(u1 - u0, 2)
            #print k
            #print sigB
            results[k] = sigB

    #print results
    argmax = max(results.items(), key=lambda x: x[1])
    return argmax[0]-1

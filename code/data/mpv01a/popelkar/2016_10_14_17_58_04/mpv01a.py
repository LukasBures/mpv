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

    histogram = {}

    pixelsY = len(gsimg)
    pixelsX = len(gsimg[0])
    sumPixels = pixelsX * pixelsY

    for tmp in range(0, 256, 1):
        histogram[tmp] = 0

    for i, val in enumerate(gsimg):
        for j, val2 in enumerate(val):
            histogram[val2] += 1

    #print histogram

    results = {}

    for threshold in range(0, 255, 1):
        # threshold = 128
        vahaBegin = 0
        vahaEnd = 0

        Ebegin = 0
        Eend = 0

        for i in range(0, threshold, 1):
            # print str(i) + ", " +str(histogram[i])
            vahaBegin += histogram[i]
            Ebegin += i * histogram[i]
        for i in range(threshold, 256, 1):
            # print str(i) + ", " + str(histogram[i])
            vahaEnd += histogram[i]
            Eend += i * histogram[i]

        if vahaBegin != 0 and vahaEnd != 0:
            # print "ebegin "+str(Ebegin)
            # print "eEnd "+str(Eend)
            Ebegin = Ebegin / vahaBegin
            Eend = Eend / vahaEnd
            # print ""

            meanBeg = vahaBegin / sumPixels
            meanEnd = vahaEnd / sumPixels

            citatelVar = 0
            for i in range(0, threshold, 1):
                citatelVar += ((i - Ebegin) * (i - Ebegin)) * histogram[i]
            var = citatelVar / meanBeg

            citatelVarEnd = 0
            for i in range(threshold, 256, 1):
                citatelVarEnd += ((i - Eend) * (i - Eend)) * histogram[i]
            varEnd = citatelVarEnd / meanEnd

            # print var
            # print varEnd

            result = (var * meanBeg) + (varEnd * meanEnd)
            # print result
            results[threshold] = result
            # print "result"

    #print results

    minim = min(results.items(), key=lambda x: x[1])
    #print minim[0]

    return minim[0]

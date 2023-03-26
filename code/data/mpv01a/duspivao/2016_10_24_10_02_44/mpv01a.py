# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 16:12:04 2016

@author: Ondrej Duspiva
"""
#
# import cv2
# import matplotlib.pyplot as plt


def make_hist(img):
    width, height = img.size
    pic = img.load()
    result = list()
    scale = list()
    for i in range(256):
        result.append(0)
        scale.append(i)

    for i in range(width):
        for j in range(height):
            result[pic[i, j]] += 1
    plt.plot(scale, result)
#    plt.show()
    return result

def otsu(imggs):
    hist = make_hist(imggs)
    max = 0.0
    maxK = 0
    N = 0.0
    ut = 0.0
    for i in range(len(hist)):
        N += hist[i]
    for i in range(len(hist)):
        ut += i*(hist[i]/N)
    for k in range(256):
        sigma2 = 0.0
        w = 0.0
        u = 0.0
        for i in range(k):
            w += (hist[i]/N)
        for i in range(k):
            if w != 0 and w != 1:
                u += i*((hist[i]/N)/w)
        w1 = 1.0-w
        if(1-w)!=0:
            u1 = (ut-u)/(1-w)
        else:
            u1 = u
        sigmaB = w*w1*((u1-u)**2)
        sigmaT = 0.0
        for q in range(len(hist)):
            sigmaT += ((i-ut)**2)*(hist[i]/N)
        if sigmaT != 0:
            sigma = sigmaB/sigmaT

            if sigma > max:
                max = sigma
                maxK = k
    return maxK
# h = make_hist(img)
# treshold = otsu(h)

# print "treshold=%d"%treshold
# plt.axvline(treshold)
# plt.show()
# cv2.waitKey(0)
# cv2.destroyAllWindows()
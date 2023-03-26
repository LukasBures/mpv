from __future__ import division
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 16:12:04 2016

@author: ondrej
"""
#
# import cv2
# import matplotlib.pyplot as plt
#
#
# img = cv2.imread("../sampleData/syntheticImg_0.jpg", cv2.COLOR_BGR2GRAY)


def make_hist(img):
    height = len(img)
    width = len(img[0])
    pic = img
    result = list()
    scale = list()
    for i in range(256):
        result.append(0)
        scale.append(i)

    for i in range(width):
        for j in range(height):
            result[pic[i, j]] += 1
    for i in range(len(result)):
        result[i] = 1.*result[i]/(width*height)
    # plt.plot(scale, result)
    # plt.show()
    return result

def otsu(img):
    temp = list()
    hist = make_hist(img)
    max = 0.0
    maxK = 0
    N = 0.0
    ut = 0.0
    for i in range(len(hist)):
        N += hist[i]
    for i in range(len(hist)):
        ut += i*(hist[i]/N)
    for k in range(256):
        # sigma2 = 0.0
        w = 0.0
        u = 0.0
        for i in range(k):
            w += (hist[i]/N)
        for i in range(0, k):
            if w != 0 and w != 1:
                # u += (i*(hist[i]/N))/w
                u+= i*hist[i]
        print k
        print w
        print u
        w1 = 1.0-w
        if (1-w)!= 0:
            u1 = (ut-u)/(1-w)
        else:
            u1 = 0

        # sigmaB = w*w1*((u1-u)**2.0)
        if w != 1 and w != 0:
            sigmaB = ((ut*w - u)**2)/(w*(1-w))
        else:
            sigmaB = 0
        # print sigmaB
        # print "--------------------------"
        if sigmaB > max:
            max = sigmaB
            maxK = k

    #     sigmaT = 0.0
    #     for q in range(len(hist)):
    #         sigmaT += ((i-ut)**2.0)*(1.*hist[i]/1.0 * N)
    #     if sigmaT != 0:
    #         sigma = 1.0 * sigmaB/(sigmaT*1.0)
    #         temp.append(sigma)
    #         if sigma > max:
    #             max = sigma
    #             maxK = k
    return maxK-1

# treshold = otsu(img)
#
# print treshold
# plt.axvline(treshold)
# plt.show()
# cv2.waitKey(0)
# cv2.destroyAllWindows()
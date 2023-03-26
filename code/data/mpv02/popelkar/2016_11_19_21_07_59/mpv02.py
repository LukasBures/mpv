from __future__ import division

# -*- coding: utf-8 -*-
import cv2
import numpy as np
from math import sqrt
import sys
#
# trainImgs = []
# trainClasses = []
#
# for i in range(0, 5):
#     for j in range(0,10):
#         trainImgs.append(cv2.imread('train/'+str(i) + '/' + str(j) + '.jpg', 0))
#         trainClasses.append(i)
#
# testImg = []
# testImg.append(cv2.imread('test/0039.jpg', 0))
# testImg.append(cv2.imread('test/0040.jpg', 0))
# testImg.append(cv2.imread('test/0027.jpg', 0))
# testImg.append(cv2.imread('test/0030.jpg', 0))
# testImg.append(cv2.imread('test/0031.jpg', 0))
# testImg.append(cv2.imread('test/0032.jpg', 0))
# testImg.append(cv2.imread('test/0033.jpg', 0))
# testImg.append(cv2.imread('test/0034.jpg', 0))
# testImg.append(cv2.imread('test/0035.jpg', 0))
# #
def kMeans(descriptors,numWords):

    words_begin = np.empty((numWords,128))

    for i in range(numWords):
        words_begin[i] = descriptors[i]

    tmpPocetInteraci = 0

    while True:

        distances_descriptors = np.zeros(len(descriptors))
        for i in range(len(descriptors)):

            distance = np.argmin(np.sum(np.power(np.subtract(words_begin, descriptors[i]), 2), 1))
            distances_descriptors[i] = distance

        words_iterate = np.zeros((numWords, 128))
        tmpPocet = 0

        for i in range(numWords):
            for j in range(len(descriptors)):
                if distances_descriptors[j] == i:
                    words_iterate[i] += descriptors[j]
                    tmpPocet += 1
            words_iterate[i] = (words_iterate[i]/tmpPocet)

            tmpPocet = 0

        if (words_begin == words_iterate).all():
            #print "pocet iteraci: " + str(tmpPocetInteraci)
            break
        else:
            words_begin = words_iterate
            tmpPocetInteraci += 1

    return words_iterate, distances_descriptors

def normalize_histogram(histogram):
    devide = sum(histogram)
    for i in range(len(histogram)):
        histogram[i] /= devide
    return histogram

def countDistance(hist1, hist2):

    temp = 0
    s1 = 0
    s2 = 0

    for i in range(len(hist1)):
        temp += hist1[i]*hist2[i]
        s1 += pow(hist1[i],2)
        s2 += pow(hist2[i],2)
    size1 = s1**(.5)
    size2 = s2**(.5)

    uhel = np.arccos((temp/(size2*size1)))
    return uhel

def bow(train_data, train_label, test_data, n_visual_words):

    trainClassesDes = []

    classesPocet = 0
    sift = cv2.xfeatures2d.SIFT_create(125)
    sift_desc = []

    for i in range(0, len(train_data)):
        _, d1 = sift.detectAndCompute(train_data[i], None)

        for j in range(len(d1)):
            sift_desc.append(d1[j])
            trainClassesDes.append(train_label[i])
        for i in range(len(train_label)):
            if classesPocet < train_label[i]:
                classesPocet = train_label[i]
    classesPocet += 1
    #
    # print "pocet trid: " + str(classesPocet)
    # print "pocet slov: " + str(n_visual_words)

    #words = kMeans(sift_desc,n_visual_words,classesPocet, trainClassesDes)
    words, allClusters = kMeans(sift_desc, n_visual_words)
    #
    # histogram = [0] * n_visual_words
    # for i in range(classesPocet):
    #     for j in range(len(sift_desc)):
    #         if trainClassesDes[j] == i:
    #             distance = np.argmin(np.sum(np.power(np.subtract(words_iterate, sift_desc[j]), 2), 1))
    #             histogram[distance] += 1
    #     # print histogram
    #     histogram = normalize_histogram(histogram)
    #     histograms.append(histogram)
    #     # print histogram
    #     histogram = [0] * n_visual_words
    # print histograms
    histograms = np.zeros((classesPocet, n_visual_words))
    histogramsCount = np.zeros(classesPocet)

    # v teto casti pocitame relativni histogramy
    for i in range(len(sift_desc)):
        histograms[trainClassesDes[i]][int(allClusters[i])] += 1
        histogramsCount[trainClassesDes[i]] += 1
    #print histograms
    for i in range(classesPocet):
        for j in range(n_visual_words):
            histograms[i][j] = histograms[i][j] / histogramsCount[i]

    cls = []


    for i in range(len(test_data)):
        _, desTest = sift.detectAndCompute(test_data[i], None)
        sift_test = []

        for i in range(len(desTest)):
            sift_test.append(desTest[i])

        histogram_test = [0] * n_visual_words
        for j in range(classesPocet):
            #distance = np.zeros((n_visual_words))
            for i in range(len(sift_test)):
                #distance[j] = np.sum(np.power((words[j] - sift_test[i], 2), 1))
                #distance = np.argmin(distance)
                distance = np.argmin(np.power((np.sum(np.power(np.subtract(words, sift_test[i]), 2), 1)), 1./2.))
                #xxxxx = np.sum(np.power(np.subtract(words, sift_test[i]), 2), 1)
                #yyy = np.subtract(words, sift_test[i])
                #zzz = np.power(np.subtract(words, sift_test[i]), 2)
                #print ""
             #   if distance == j:
                histogram_test[distance] += 1

        normalized_hists = histograms
        normalized_test =  normalize_histogram(histogram_test)

        #print "___ test: " + str(normalized_test)

        tmpMin = sys.maxint
        tmpIdx = sys.maxint

        for i in range(len(normalized_hists)):
            actMin = countDistance(normalized_hists[i],normalized_test)
            if  actMin < tmpMin:
                tmpMin = actMin
                tmpIdx = i

        cls.append(tmpIdx)

    return cls

#print bow(trainImgs, trainClasses, testImg, 6)

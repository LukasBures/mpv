from __future__ import division
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from math import sqrt
import sys

trainClassesDes = []
histograms = []

trainImgs = []
trainClasses = []
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

def kMeans(descriptors,numWords,classesPocet):

    words_begin = np.empty((numWords,128))

    for i in range(numWords):
        words_begin[i] = descriptors[i]

    tmpPocetInteraci = 0
    print words_begin
    while True:

        distances_descriptors = np.empty((1,len(descriptors)))

        for i in range(len(descriptors)):

            distance = np.argmin(np.sum(np.power(np.subtract(words_begin, descriptors[i]), 2), 1))
            distances_descriptors[0][i] = distance

        words_iterate = np.empty((numWords, 128))
        sumOf = np.zeros((1,128))
        tmpPocet = 0

        for i in range(numWords):
            for j in range(len(descriptors)):
                if distances_descriptors[0][j] == i:
                    sumOf += descriptors[j]
                    tmpPocet += 1
            words_iterate[i] = (sumOf/tmpPocet)[0]
            sumOf = np.zeros((1,128))
            tmpPocet = 0

        if (words_begin != words_iterate).all():
            words_begin = words_iterate
            tmpPocetInteraci += 1
        else:
            #print "pocet iteraci: " + str(tmpPocetInteraci)
            histogram = [0] * numWords
            for i in range(classesPocet):
                for j in range(len(descriptors)):
                    if trainClassesDes[j] == i:
                        distance = np.argmin(np.sum(np.power(np.subtract(words_iterate, descriptors[j]), 2), 1))
                        histogram[distance] += 1
                histogram = normalize_histogram(histogram)
                histograms.append(histogram)
                print histogram
                histogram = [0] * numWords
            return words_iterate
            break

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
    size1 = sqrt(s1)
    size2 = sqrt(s2)

    uhel = np.arccos((temp/(size2*size1)))
    return uhel

def bow(train_data, train_label, test_data, n_visual_words):


    classesPocet = 0
    sift = cv2.xfeatures2d.SIFT_create(125)
    sift_desc = []

    for i in range(0, len(train_data)):
        _, d1 = sift.detectAndCompute(train_data[i].astype(np.uint8), None)

        for j in range(len(d1)):
            if j < 125:
                sift_desc.append(d1[j])
                trainClassesDes.append(train_label[i])
        for i in range(len(train_label)):
            if classesPocet < train_label[i]:
                classesPocet = train_label[i]
    classesPocet += 1

    print "pocet trid: " + str(classesPocet)
    print "pocet slov: " + str(n_visual_words)

    words = kMeans(sift_desc,n_visual_words,classesPocet)
    cls = []
    for i in range(len(test_data)):
        _, desTest = sift.detectAndCompute(test_data[i], None)
        sift_test = []

        for i in range(len(desTest)):
            sift_test.append(desTest[i])

        histogram_test = [0] * n_visual_words
        for j in range(classesPocet):
            for i in range(len(sift_test)):
                distance = np.argmin(np.sum(np.power(np.subtract(words, sift_test[i]), 2), 1))
                if distance == j:
                    histogram_test[distance] += 1

        normalized_hists = histograms
        normalized_test =  normalize_histogram(histogram_test)

        tmpMin = sys.maxint
        tmpIdx = sys.maxint

        for i in range(len(normalized_hists)):
            actMin = countDistance(normalized_hists[i],normalized_test)
            if  actMin < tmpMin:
                tmpMin = actMin
                tmpIdx = i

        cls.append(tmpIdx)

    return cls

#print bow(trainImgs, trainClasses, testImg, 5)

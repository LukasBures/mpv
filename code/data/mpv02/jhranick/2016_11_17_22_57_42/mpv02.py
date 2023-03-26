# -*- coding: utf-8 -*-
"""
Semester Project MPV - 2
Training and using Bag of Words algorithm for image classification
@author: Bc. Jan Hranicka
@email: jhranick@students.zcu.cz
@version: 1.5.0
"""

import logging
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from os import path
from numpy.linalg import norm
from random import shuffle
from datetime import datetime


# Logging utility initialization
logger = logging.getLogger('mpv02')
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

logger.addHandler(ch)


def pdist2(a, b):
    return np.sqrt(np.sum(np.power(np.subtract(a, b), 2)))


def nearestneighbor(data, cnts):
    distances = np.zeros((len(data), len(cnts)))
    for n in range(len(cnts)):
        for i in range(len(data)):
            distances[i, n] = pdist2(data[i], cnts[n])
    classification = distances.argmin(1)
    return classification


def kmeans(data, num_clusters, threshold=0.001):
    rand_indices = np.arange(0, len(data))
    np.random.shuffle(rand_indices)
    centroids = data[rand_indices[:num_clusters]]

    criterion = 0
    classification = None
    it = 0
    while True:
        distances = np.zeros((len(data), num_clusters))
        for n in range(num_clusters):
            for i in range(len(data)):
                distances[i, n] = pdist2(data[i], centroids[n])**2

        classification = distances.argmin(1)
        for c in range(num_clusters):
            pom = data[np.where(classification == c)]
            newMu = np.sum(pom, axis=0, dtype=np.float32) / len(pom)
            centroids[c] = newMu

        criterion_prev = criterion
        criterion = distances.sum(0).sum()
        if criterion == criterion_prev or abs(criterion - criterion_prev) < threshold or it > 20:
            logger.debug('Automatic break after 20 iterations' if it > 20 else 'Done')
            break
        else:
            it += 1
            logger.debug('New Iteration [%s]' % it)

    return classification, centroids


def bow(train_data, train_label, test_data, n_visual_words):
    """
    Provede klasifikaci pomoci BoW algoritmu.

    :param train_data: Vstupni list trenovacich sedotonovych obrazku.
    :rtype train_data: list of 2D ndarray, uint8

    :param train_label: Vstupni list trid, do kterych patri trenovaci sedotonove obrazky.
    :rtype train_label: list of int

    :param test_data: Vstupni list testovacich sedotonovych obrazku.
    :rtype test_data: list of 2D ndarray, uint8

    :param n_visual_words: Pocet vizualnich slov.
    :rtype n_visual_words: int

    :return cls: Vystupni list trid odpovidajicich obrazkum v test_data.
    :rtype cls: list of int
    """
    logger.debug('STARTING COMPUTATION')
    cmp_start = datetime.now()
    labels = {0: "cloth", 1: "coffee", 2: "basket", 3: "brick", 4: "metal"}

    numClasses = len(np.bincount(train_label))
    sift = cv2.xfeatures2d.SIFT_create(125)

    descriptors = list()
    class_labels = list()
    for i in range(len(train_data)):
        (kps, descs) = sift.detectAndCompute(train_data[i], None)
        descriptors.append(descs)
        class_labels.append(np.full(len(descs), train_label[i], dtype=np.uint8))

    descriptors = np.vstack(descriptors)
    class_labels = np.hstack(class_labels)
    logger.debug('Data preprocessed - Total number of descriptors [%s]' % len(descriptors))
    logger.debug('Computation time: %s seconds' % (datetime.now() - cmp_start))
    # exit(2)

    cmp_start_2 = datetime.now()
    (classif, visual_words) = kmeans(descriptors, n_visual_words)

    logger.debug('K-means clustering finished - Total number of clusters [%s]' % len(visual_words))
    logger.debug('Computation time: %s seconds' % (datetime.now() - cmp_start_2))

    cmp_start_3 = datetime.now()
    class_histograms = list()
    for i in range(numClasses):
        pom = classif[class_labels == i]
        hist = np.bincount(pom, minlength=n_visual_words)
        hist_norm = hist/hist.sum(dtype=np.float32)

        class_histograms.append(hist_norm)

    logger.debug('Classes histograms created')
    logger.debug('Computation time: %s seconds' % (datetime.now() - cmp_start_3))
    logger.debug(class_histograms)


    # Processing test images
    cls = list()
    for iid, img in enumerate(test_data):
        (kps, descs) = sift.detectAndCompute(img, None)
        cnts = nearestneighbor(descs, visual_words)

        hist = np.bincount(cnts, minlength=n_visual_words)
        hist_norm = hist / hist.sum(dtype=np.float32)

        alpha = np.zeros(numClasses)
        for i in range(numClasses):
            u = class_histograms[i]
            v = hist_norm
            alpha[i] = np.arccos(np.dot(u, v) / (norm(u) * norm(v)))
        cl = alpha.argmin()
        cls.append(cl)

        # logger.debug('Recognized image: %s' % labels[cl])
        # plt.imshow(img, cmap='gray')
        # plt.show()
        # break

    logger.debug('Image classification finished - Total number of testing images [%s]' % len(test_data))
    logger.debug('Computation time: %s seconds' % (datetime.now() - cmp_start))
    return cls


if __name__ == '__main__':
    imgPath = r"D:\OneDrive\Faculty of Applied Science\Postgraduate Master Studies\2016-2017\MPV\Semester Projects Repository\SP2\train"
    imgsTrain = list()
    trainLabel = list()
    for i in range(5):
        files = path.join(imgPath, str(i))
        for f in os.listdir(files)[:24]:  # 24 instead of 4 (!!!)
            f = path.join(files, f)
            imgsTrain.append(cv2.imread(f, cv2.IMREAD_GRAYSCALE))
            trainLabel.append(i)
    logger.debug(trainLabel)

    labels = [0,0,0,1,1,1,2,2,2,3,3,3,3,4,4,4,4]
    testPath = r"D:\OneDrive\Faculty of Applied Science\Postgraduate Master Studies\2016-2017\MPV\Semester Projects Repository\SP2\train\test"
    imgsTestPaths = [path.join(testPath, f) for f in os.listdir(testPath)]

    labeldict = dict()
    for i in range(len(labels)):
        labeldict[imgsTestPaths[i]] = labels[i]

    shuffle(imgsTestPaths)
    imgsTest = list()
    labels = list()
    labels_shuffled = list()
    for pth in imgsTestPaths:
        labels_shuffled.append(labeldict[pth])
        imgsTest.append(cv2.imread(pth, cv2.IMREAD_GRAYSCALE))

    cls = bow(imgsTrain, trainLabel, imgsTest, 10)
    logger.debug('Input: %s' % labels_shuffled)
    logger.debug('Result: %s' % cls)
    logger.debug('Number of bad classified: %s' % (np.array(labels_shuffled) != np.array(cls)).sum())

    # kmeans(data, 2)

    # img = r"D:\OneDrive\Faculty of Applied Science\Postgraduate Master Studies\2016-2017\MPV\Semester Projects Repository\SP2\train\0\0001.jpg"
    # img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    # bow(img, 0, None, 5)

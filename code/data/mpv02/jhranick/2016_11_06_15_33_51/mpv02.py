# -*- coding: utf-8 -*-
"""
Semester Project MPV - 2

@author: Bc. Jan Hranicka
@email: jhranick@students.zcu.cz
@version: 1.0.0
"""

import logging
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from os import path
from collections import defaultdict
from numpy.linalg import norm
from random import shuffle


# Logging utility initialization
logger = logging.getLogger('mpv02')
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

logger.addHandler(ch)


def pdist2(a, b):
    return np.sqrt(np.sum(np.power(a-b, 2)))


def nearestneighbor(data, cnts):
    distances = np.zeros((len(data), len(cnts)))
    for n in range(len(cnts)):
        for i in range(len(data)):
            # logger.debug(pdist2(data[i], cnts[n]))
            distances[i, n] = pdist2(data[i], cnts[n])
    # logger.debug(distances)

    classification = distances.argmin(1)
    # logger.debug(classification)

    # plt.axis([-1, 5, -1, 5])
    # colors = ['ro', 'bo']
    # for i in range(len(data)):
    #     plt.plot(data[i, 0], data[i, 1], colors[classification[i]])
    # plt.plot(cnts[:, 0], cnts[:, 1], 'gx')
    # plt.show()
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
                distances[i, n] = pdist2(data[i], centroids[n])

        # logger.debug('\n%s' % distances)
        classification = distances.argmin(1)
        # logger.debug(classification)

        # colors = ['ro', 'bo']
        # plt.axis([-1, 5, -1, 5])
        # for c in range(num_clusters):
        #     pom = data[np.where(classification == c)]
        #     newMu = np.sum(pom, axis=0, dtype=np.float32) / len(pom)
        #     centroids[c] = newMu
        #     plt.plot(pom[:, 0], pom[:, 1], colors[c])
        #     plt.plot(centroids[c, 0], centroids[c, 1], 'gx')
        # plt.show()

        criterion_prev = criterion
        criterion = distances.sum(0).sum()
        if criterion == criterion_prev or abs(criterion - criterion_prev) < threshold or it > 20:
            if it > 20:
                logger.debug('Broken down after 20 iterations')
            else:
                logger.debug('Done')
            # plt.show()
            break
        else:
            logger.debug('New Iteration')
            it += 1

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
    # labels = {0: "cloth", 1: "coffee", 2: "basket", 3: "brick", 4: "metal"}

    numClasses = len(np.bincount(train_label))
    sift = cv2.xfeatures2d.SIFT_create(125)
    lengths = np.zeros(numClasses)
    hist_data = np.zeros((numClasses, n_visual_words))
    class_descriptors = defaultdict(list)

    for i in range(len(train_data)):
        (kps, descs) = sift.detectAndCompute(train_data[i], None)
        # logger.debug('Number of descriptors: %s' % len(descs))

        lengths[trainLabel[i]] += len(descs)
        class_descriptors[train_label[i]].append(descs)

    centroids_all = list()
    dhist_all = list()
    for cl, dlist in class_descriptors.items():
        data = np.concatenate(np.array(dlist))
        (classif, centroids) = kmeans(data, n_visual_words)
        for c in centroids:
            centroids_all.append(c)
        logger.debug(classif)

        hist_data[train_label[cl]] = np.bincount(classif)
        dhist = hist_data[train_label[cl]]/lengths[cl]
        dhist_all.append(dhist)
    # logger.debug('Number of centroids: %s' % len(centroids_all))


    # Processing test image
    cls = list()
    for iid, img in enumerate(test_data):

        (kps, descs) = sift.detectAndCompute(img, None)
        # logger.debug('Number of descriptors: %s' % len(descs))

        # logger.debug(len(np.array(centroids_all)))
        cnts = nearestneighbor(descs, np.array(centroids_all))

        cnts_data = np.bincount(cnts, minlength=len(centroids_all))

        cdhist = cnts_data/float(cnts_data.sum())
        hists = np.split(cdhist, numClasses)


        class_hists = np.split(np.array(dhist_all), numClasses)
        results = list()
        alpha = np.zeros(numClasses)
        for i in range(numClasses):
            if hists[i].sum() == 0:
                alpha[i] = 10
                continue
            u = class_hists[i]
            v = hists[i]
            alpha[i] = np.arccos(np.dot(u, v)/(norm(u)*norm(v)))
            # logger.debug('Alpha for %s. class = %s' % (i, alpha[i]))
            # pass

        # logger.debug('Recognized image: %s' % labels[alpha.argmin()])
        cls.append(alpha.argmin())
        # plt.imshow(img, cmap='gray')
        # plt.show()
    return cls


if __name__ == '__main__':
    data = np.array([[1, 3], [1, 4], [2, 3], [2, 4], [3, 1], [4, 1], [4, 2]], dtype=np.float32)
    # (classif, centroids) = kmeans_test(data, 2)
    # test_data = np.array([[3, 2], [4, 0], [2, 3.5], [0, 4]])
    # cls = knn(test_data, centroids)
    #
    # plt.axis([-1, 5, -1, 5])
    # colors = ['ro', 'bo']
    # cnt_colors = ['rx', 'bx']
    # test_colors = ['rs', 'bs']
    # for i in range(len(data)):
    #     plt.plot(data[i, 0], data[i, 1], colors[classif[i]])
    #
    # for i in range(len(centroids)):
    #     plt.plot(centroids[i, 0], centroids[i, 1], cnt_colors[i])
    # for i in range(len(test_data)):
    #     plt.plot(test_data[i, 0], test_data[i, 1], test_colors[cls[i]])
    #     plt.plot([centroids[cls[i], 0], test_data[i, 0]], [centroids[cls[i], 1], test_data[i, 1]], 'g')
    #
    #
    # plt.show()
    # exit(13)

    imgPath = r"D:\OneDrive\Faculty of Applied Science\Postgraduate Master Studies\2016-2017\MPV\Semester Projects Repository\SP2\train"
    imgsTrain = list()
    trainLabel = list()
    for i in range(5):
        files = path.join(imgPath, str(i))
        for f in os.listdir(files)[:20]:
            f = path.join(files, f)
            imgsTrain.append(cv2.imread(f, cv2.IMREAD_GRAYSCALE))
            trainLabel.append(i)
    logger.debug(trainLabel)

    testPath = r"D:\OneDrive\Faculty of Applied Science\Postgraduate Master Studies\2016-2017\MPV\Semester Projects Repository\SP2\train\test"
    imgsTestPaths = [path.join(testPath, f) for f in os.listdir(testPath)]
    shuffle(imgsTestPaths)
    imgsTest = list()
    for pth in imgsTestPaths:
        imgsTest.append(cv2.imread(pth, cv2.IMREAD_GRAYSCALE))

    cls = bow(imgsTrain, trainLabel, imgsTest, 7)
    logger.debug(cls)

    # kmeans(data, 2)

    # img = r"D:\OneDrive\Faculty of Applied Science\Postgraduate Master Studies\2016-2017\MPV\Semester Projects Repository\SP2\train\0\0001.jpg"
    # img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    # bow(img, 0, None, 5)

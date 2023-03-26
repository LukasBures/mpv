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
    # centroids = np.array([[0, 5], [1, 5], [3, 1]])

    criterion = 0
    classification = None
    it = 0
    while True:
        distances = np.zeros((len(data), num_clusters))
        for n in range(num_clusters):
            for i in range(len(data)):
                distances[i, n] = pdist2(data[i], centroids[n])**2

        # logger.debug('\n%s' % distances)
        classification = distances.argmin(1)
        # logger.debug(classification)

        # colors = ['ro', 'bo', 'co']
        # plt.axis([-1, 4, -1, 6])
        for c in range(num_clusters):
            pom = data[np.where(classification == c)]
            newMu = np.sum(pom, axis=0, dtype=np.float32) / len(pom)
            centroids[c] = newMu
            # plt.plot(pom[:, 0], pom[:, 1], colors[c])
            # plt.plot(centroids[c, 0], centroids[c, 1], 'gx')
        # plt.show()

        criterion_prev = criterion
        criterion = distances.sum(0).sum()
        if criterion == criterion_prev or abs(criterion - criterion_prev) < threshold or it > 20:
            if it > 20:
                logger.debug('Automatic break after 20 iterations')
            else:
                logger.debug('Done')
            # plt.show()
            break
        else:
            logger.debug('New Iteration')
            it += 1

    return classification, centroids


def kmeans_bin(data, num_clusters, threshold=0.001):
    rand_indices = np.arange(0, len(data))
    np.random.shuffle(rand_indices)
    centroids = data[rand_indices[:2]]  # Get first two random indices only
    # centroids = np.array([[0, 5], [1, 5], [3, 1]])

    criterion = 0
    act_clusters = 2
    classification = None
    it = 0
    while True:
        distances = np.zeros((len(data), act_clusters))
        for n in range(act_clusters):
            for i in range(len(data)):
                distances[i, n] = pdist2(data[i], centroids[n])**2

        # logger.debug('\n%s' % distances)
        classification = distances.argmin(1)
        # logger.debug(classification)

        # colors = ['ro', 'bo', 'co']
        # plt.axis([-1, 4, -1, 6])
        # logger.debug(distances)
        for c in range(act_clusters):
            pom = data[np.where(classification == c)]
            newMu = np.sum(pom, axis=0, dtype=np.float32) / len(pom)
            centroids[c] = newMu
            # plt.plot(pom[:, 0], pom[:, 1], colors[c])
            # plt.plot(centroids[c, 0], centroids[c, 1], 'gx')
        # plt.show()

        if act_clusters < num_clusters:
            mins = np.argmin(distances, 1)
            # logger.debug(distances)

            J = np.zeros(act_clusters)
            maxs = np.zeros(act_clusters)
            argmaxs = np.zeros(act_clusters, dtype=np.uint8)
            for col in range(act_clusters):
                logger.warning(distances[mins == col])
                J[col] = distances[mins == col].sum(0)[col]             # working ?
                # logger.debug(distances[mins == col].shape)
                # logger.debug('%s ----- %s' % (col, distances[mins == col].max(0)))
                maxs[col] = distances[mins == col].max(0)[col]
                # logger.debug('Maximum for column %s in %s' % (col, maxs[col]))
                # logger.debug('Index of max: %s' % np.where(distances[:,col]==maxs[col]))
                argmaxs[col] = np.where(distances[:, col] == maxs[col])[0][0]
            # logger.debug('Criterial function:\n%s' % J)
            # logger.debug('Arguments of maximums are:\n%s' % argmaxs)
            # logger.debug('\n%s' % distances)
            # logger.debug(J)
            # exit(8)

            # logger.debug('Bude rozdelen shluk %s s J=%s' % (np.argmax(J), J.max()))
            # # logger.debug(distances[argmaxs[J.argmax()], J.argmax()])
            # logger.debug('Novy prvek je ve sloupci %s s hodnotou %s' % (J.argmax(), distances[argmaxs[J.argmax()], J.argmax()]))
            # logger.debug('Novy stred ma index=%s a souradnice: %s' % (
            #     argmaxs[J.argmax()], data[argmaxs[J.argmax()], :]))
            addedMu = data[argmaxs[J.argmax()], :]

            # plt.axis([-1, 4, -1, 6])
            # plt.plot(data[:, 0], data[:, 1], 'kx')
            # plt.plot(centroids[:, 0], centroids[:, 1], 'go')
            # plt.plot(addedMu[0], addedMu[1], 'mo')
            # plt.show()
            # exit(5)
            act_clusters += 1
            # logger.debug(centroids)
            centroids = np.vstack((centroids, addedMu))
            # logger.debug(centroids)
            continue


            # J = distances.sum(0)
            # logger.debug(J)
            # logger.debug('Bude rozdelen shluk %s s J=%s' % (np.argmax(J), J.max()))
            #
            # # Select the most distant vector in class with max(J)
            # indices = np.where(~distances.all(axis=1))[0]
            # # logger.debug(distances)
            # distances[indices] = np.zeros(act_clusters)
            # logger.debug(distances)
            # logger.debug('Novy prvek je ve sloupci %s s hodnotou %s' % (np.argmax(J), distances[:, np.argmax(J)].max()))
            # logger.debug('Novy stred ma index=%s a souradnice: %s' % (distances[:, np.argmax(J)].argmax(), data[distances[:, np.argmax(J)].argmax()]))
            # addedMu = data[distances[:, np.argmax(J)].argmax()]
            #
            # plt.axis([-1, 4, -1, 6])
            # plt.plot(data[:, 0], data[:, 1], 'kx')
            # plt.plot(centroids[:, 0], centroids[:, 1], 'go')
            # plt.plot(addedMu[0], addedMu[1], 'mo')
            # plt.show()
            # exit(3)

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
    labels = {0: "cloth", 1: "coffee", 2: "basket", 3: "brick", 4: "metal"}

    numClasses = len(np.bincount(train_label))
    sift = cv2.xfeatures2d.SIFT_create(125)

    for i in range(len(train_data)):
        (kps, descs) = sift.detectAndCompute(train_data[i], None)

        if i == 0:
            descriptors = descs
            class_labels = np.full((1, len(descs)), train_label[i], dtype=np.uint8)
        else:
            descriptors = np.append(descriptors, descs, axis=0)
            class_labels = np.append(class_labels, np.full((1, len(descs)), train_label[i], dtype=np.uint8))

    logger.debug('Data preprocessed - Total number of descriptors [%s]' % len(descriptors))

    (classif, centroids) = kmeans(descriptors, n_visual_words)

    class_histograms = list()
    for i in range(numClasses):
        pom = classif[class_labels == i]
        hist = np.bincount(pom, minlength=n_visual_words)
        hist_norm = hist/hist.sum(dtype=np.float32)
        # print(hist_norm)

        class_histograms.append(hist_norm)
        # plt.hist(pom, bins=np.arange(0, n_visual_words+1), normed=True)
        # plt.show()


    # Processing test images
    cls = list()
    for iid, img in enumerate(test_data):
        (kps, descs) = sift.detectAndCompute(img, None)
        cnts = nearestneighbor(descs, centroids)

        hist = np.bincount(cnts, minlength=n_visual_words)
        hist_norm = hist / hist.sum(dtype=np.float32)
        # print(hist_norm)

        # plt.hist(cnts, bins=np.arange(0, n_visual_words + 1), normed=True)
        # plt.show()

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
    return cls


if __name__ == '__main__':
    data = np.array([[0, 0], [1, 0], [1, 1], [2, 1], [3, 1], [3, 2], [0, 4], [0, 5], [1, 4], [1, 5]], dtype=np.float32)
    # (classif, centroids) = kmeans_bin(data, 7)
    # # test_data = np.array([[3, 2], [4, 0], [2, 3.5], [0, 4]])
    # # cls = knn(test_data, centroids)
    # #
    # plt.axis([-1, 4, -1, 6])
    # colors = ['ro', 'bo', 'co', 'mo', 'yo', 'go', 'ko']
    # cnt_colors = ['rx', 'bx', 'cx', 'mx', 'yx', 'gx', 'kx']
    # # test_colors = ['rs', 'bs']
    # for i in range(len(data)):
    #     plt.plot(data[i, 0], data[i, 1], colors[classif[i]])
    # #
    # for i in range(len(centroids)):
    #     plt.plot(centroids[i, 0], centroids[i, 1], cnt_colors[i])
    # # for i in range(len(test_data)):
    # #     plt.plot(test_data[i, 0], test_data[i, 1], test_colors[cls[i]])
    # #     plt.plot([centroids[cls[i], 0], test_data[i, 0]], [centroids[cls[i], 1], test_data[i, 1]], 'g')
    # #
    # #
    # plt.show()
    # exit(13)

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

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 13:38:32 2014

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits: 
@version: 3.0.0

Revision Note:
3.0.0 - 7.11.2016 - Updated for OpenCV 3.1.0 version
"""

import numpy as np
import cv2


# ----------------------------------------------------------------------------------------------------------------------
def calculate_sift_descriptors(train_data, train_label):
    sift = cv2.xfeatures2d.SIFT_create(125)
    n_class = len(set(train_label))
    n_desc_in_cls = np.zeros((1, n_class), np.int)

    print "Processing:",
    for idx, img in enumerate(train_data):
        if (idx + 1) % 10 == 0:
            print ".",

        _, des = sift.detectAndCompute(img, None)
        if idx == 0:
            all_des = des
        else:
            all_des = np.concatenate([all_des, des])
        n_desc_in_cls[0, train_label[idx]] = n_desc_in_cls[0, train_label[idx]] + des.shape[0]

    print "OK"
    return all_des, n_desc_in_cls


# ----------------------------------------------------------------------------------------------------------------------
def calculate_centroids(descriptors, n_visual_words):
    # Flag to specify the number of times the algorithm is executed using
    # different initial labellings. The algorithm returns the labels that
    # yield the best compactness
    attempts = 10
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv2.kmeans(descriptors, n_visual_words, None, criteria=criteria, attempts=attempts,
                                  flags=cv2.KMEANS_PP_CENTERS)
    
    return label, center


# ----------------------------------------------------------------------------------------------------------------------
def calculate_models(label, n_desc_in_cls, n_visual_words):
    
    model_histograms = np.zeros((n_desc_in_cls.shape[1], n_visual_words), np.float64)

    tmp_n_desc_in_cls = np.zeros((1, n_desc_in_cls.shape[1] + 1), np.int)
    for i in range(n_desc_in_cls.shape[1] + 1):
        if i == 0:
            tmp_n_desc_in_cls[0, 0] = 0
        else:
            tmp_n_desc_in_cls[0, i] = tmp_n_desc_in_cls[0, i - 1] + n_desc_in_cls[0, i - 1]
    
    for i in range(n_desc_in_cls.shape[1]):
        for j in range(tmp_n_desc_in_cls[0, i], tmp_n_desc_in_cls[0, i + 1]):
            model_histograms[i, label[j]] += 1.0
    
    model_histograms = np.float64(model_histograms) / np.float64(n_desc_in_cls.T)
    
    return model_histograms


# ----------------------------------------------------------------------------------------------------------------------
def calculate_test_histogram(img, centers):
    sift = cv2.xfeatures2d.SIFT_create(125)
    _, des = sift.detectAndCompute(img, None)
    
    des = np.float64(des)
    label = np.zeros((1, des.shape[0]), np.int)
    
    # najde nejblizsi tridu
    for i in range(des.shape[0]):
        distance = np.zeros((1, centers.shape[0]), np.float64)
        for j in range(centers.shape[0]):
            distance[0, j] = np.sqrt(np.sum(np.power(des[i] - centers[j], 2.0)))
        mi = np.min(distance)
        
        for k in range(centers.shape[0]):
            if mi == distance[0, k]:
                label[0, i] = k
                break
    
    # vypocte histogram
    test_hist = np.zeros((1, centers.shape[0]), np.float64)
    for i in range(label.shape[1]):
        test_hist[0, label[0, i]] += 1.0
    
    return test_hist / des.shape[0]


# ----------------------------------------------------------------------------------------------------------------------
def calculate_angle(model_histograms, test_hist):
    
    model_histograms = np.float64(model_histograms)
    test_hist = np.float64(test_hist)
    
    res = np.zeros((1, model_histograms.shape[0]), np.float64)
    for i in range(model_histograms.shape[0]):
        a = np.sum(model_histograms[i, :] * test_hist)
        b = np.sqrt(np.sum(np.power(model_histograms, 2)))
        c = np.sqrt(np.sum(np.power(test_hist, 2)))
        res[0, i] = np.arccos(a / (b * c))
    
    mi = np.min(res)
    cls = []
    for i in range(res.shape[1]):
        if mi == res[0, i]:
            cls = i
            break

    return cls


# ----------------------------------------------------------------------------------------------------------------------
def bow(train_data, train_label, test_data, n_visual_words):
    """

    """
    # Vypocte deskriptory a jejich pocty v ramci jedne tridy.
    train_desc, n_desc_in_class = calculate_sift_descriptors(train_data, train_label)
    
    # Vypocet vizualnich slov pomoci metody K-means, kde K = n_visual_words.
    label, centers = calculate_centroids(train_desc, n_visual_words)
    
    # Vypocita modely, ktere jsou reprezentovany histogramy.
    model_histograms = calculate_models(label, n_desc_in_class, n_visual_words)
    
    # List vyslednych klasifikaci jednotlivych obrazku.
    cls = []

    # Klasifikace obrazku.
    print "Klasifikace:",
    for idx, img in enumerate(test_data):
        if (idx + 1) % 10 == 0:
            print ".",
        # Vypocita histogram vstupniho obrazku.
        test_hist = calculate_test_histogram(img, centers)
        
        # Klasifikace.
        cls.append(calculate_angle(model_histograms, test_hist))

    print "OK"

    # Vraci pole klasifikaci.
    return cls

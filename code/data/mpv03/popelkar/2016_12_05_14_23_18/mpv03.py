# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 13:38:32 2014

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits: 
@version: 2.0.0
"""

import cv2
import numpy as np
import os
from skimage.feature import hog


def bgr_dist(img):
    """
        Provede transformaci obrazku.
    """
    imgout = np.sqrt(np.power(img[:, :, 0], 2.0) + np.power(img[:, :, 1], 2.0) + np.power(img[:, :, 2], 2.0))
    imgout = (imgout / imgout.max()) * 255
    imgout = np.uint8(imgout)
    return imgout

#def train(cls_path, n_cls):

def train(train_data, train_labels):
    """
        Nacte obrazky, spocita FV a natrenuje SVM.
    """
    print
    print "--------------------------------------------------------"
    print "Vypocet FV z trenovaci sady:"
    #train_data, train_labels = read_data(cls_path, n_cls)
    train_fv = list()
    train_label = list()

    for i,d in enumerate(train_data):
        img = bgr_dist(d)
        feat, hog_img = hog(img, orientations=8, pixels_per_cell=(64,64),cells_per_block=(8,8),visualise=True)
        train_fv.append(feat)
        train_label.append(train_labels[i])

        #cv2.imshow("HOD Img", hog_img)
        #cv2.waitKey(1)
    svm = train_svm(np.asarray(train_fv, dtype="float32"),np.asarray([train_label], dtype="int32"))

    # TODO: implementace trenovani


    return svm


def train_svm(data, responses):
    """
        Natrenuje SVM
    """

    # TODO: implementace trenovani smv

    my_svm = cv2.ml.SVM_create()
    my_svm.setType(cv2.ml.SVM_C_SVC)
    my_svm.setKernel(cv2.ml.SVM_LINEAR)

    print "trenovani svm"
    my_svm.train(data, cv2.ml.ROW_SAMPLE, responses)


    return my_svm

# ------------------------------------------------------------------------------
def classify(train_data, train_label, test_data):


    for i,d in enumerate(test_data):
        img = bgr_dist(d)
        feat, hog_img = hog(img, orientations=8, pixels_per_cell=(4,4),cells_per_block=(8,8),visualise=True)
        test_data.append(feat)

        cv2.imshow("HOD Img", hog_img)
        cv2.waitKey(1)

    train(train_data, train_label)

    svm = train_svm(np.asarray(train_data, dtype="float32"), np.asarray([train_label], dtype="int32"))

    _, predicted = svm.predict(np.asarray(test_data, dtype='float32'))
    predicted = np.squeeze(predicted)
    predicted = predicted.astype(dtype="int32")

    """
    Provede klasifikaci dat pomoci SVM klasifikatoru.

    :param train_data: Vstupni list trenovacich sedotonovych obrazku.
    :rtype train_data: list of 2D ndarray, uint8

    :param train_label: Vstupni list trid, do kterych patri trenovaci sedotonove obrazky.
    :rtype train_label: list of int

    :param test_data: Vstupni list testovacich sedotonovych obrazku.
    :rtype test_data: list of 2D ndarray, uint8

    :return cls: Vystupni list trid odpovidajicich obrazkum v test_data.
    :rtype cls: list of int
    """
    cls = predicted

    return cls

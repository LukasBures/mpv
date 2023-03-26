# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 15:47:41 2014

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


def train_svm(data, responses):
    """
        Natrenuje SVM
    """

    # TODO: implementace trenovani smv



    return my_svm


def read_data(data_path, n_cls):
    """
        Nacte obrazky
    """
    datas = []
    labels = []

    for i in range(0, n_cls):
        p = data_path + "/" + str(i) + "/"
        for fn in os.listdir(p):
            gs = cv2.imread(p + fn, cv2.CV_LOAD_IMAGE_COLOR)
            datas.append(gs)
            labels.append(i)

    return datas, labels


def train(cls_path, n_cls):
    """
        Nacte obrazky, spocita FV a natrenuje SVM.
    """
    print
    print "--------------------------------------------------------"
    print "Vypocet FV z trenovaci sady:"
    train_data, train_labels = read_data(cls_path, n_cls)
    train_fv = list()
    train_label = list()


    # TODO: implementace trenovani


    return svm


def compute_confusion(labels_gt, labels_test):
    """
        Spocita matici zamen.
    """

    # TODO: implementace matice zamen


    return confusion


def test(cls_path, n_cls, svm):
    print
    print "--------------------------------------------------------"
    print "Vypocet FV z testovaci sady:"

    test_data, test_labels = read_data(cls_path, n_cls)
    test_fv = list()
    test_label = list()




    # TODO: implementace testovani









    print
    print "--------------------------------------------------------"
    print
    predicted = svm.predict_all(np.asarray(test_fv, dtype='float32'))
    predicted = np.squeeze(predicted)

    print "--------------------------------------------------------"
    print "Vysledky klasifikace:"
    print "Chybne klasifikovano:",
    print np.sum(predicted != np.asarray(test_label, dtype='float32')),
    print "/", len(predicted)
    err = (predicted != np.asarray(test_label, dtype='float32')).mean()
    print 'Error: %.2f %%' % (err * 100)
    print "--------------------------------------------------------"
    print

    confusion = compute_confusion(test_label, list(predicted))
    print "--------------------------------------------------------"
    print "Confusion matrix:"
    print confusion
    print "--------------------------------------------------------"
    print


if __name__ == "__main__":
    n_class = 5
    clsTrainPath = "./img/train/"
    clsTestPath = "./img/test/"
    cv2.namedWindow("HOG Img", 0)

    trained_svm = train(clsTrainPath, n_class)
    test(clsTestPath, n_class, trained_svm)

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 14:47:28 2014

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
        :param img:
    """
    imgout = np.sqrt(np.power(img[:, :, 0], 2.0) + np.power(img[:, :, 1], 2.0) + np.power(img[:, :, 2], 2.0))
    imgout = (imgout / imgout.max()) * 255
    imgout = np.uint8(imgout)
    return imgout


def train_random_forest(data, responses):
    """
        Natrenuje nahodny les
        :param data:
        :param responses:
    """
    rf = cv2.RTrees()
    print "Training random forest"
    rf.train(data, cv2.CV_ROW_SAMPLE, responses)

    return rf


def read_data(data_path, n_cls):
    """
        Nacte obrazky
        :param n_cls:
        :param data_path:
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
        Nacte obrazky, spocita FV a natrenuje Random Forest.
        :param n_cls:
        :param cls_path:
    """
    print
    print "--------------------------------------------------------"
    print "Vypocet FV z trenovaci sady:"
    train_data, train_labels = read_data(cls_path, n_cls)
    train_fv = list()
    train_label = list()

    for i, d in enumerate(train_data):
        print ".",
        img = bgr_dist(d)
        feat, hog_img = hog(img, orientations=8, pixels_per_cell=(64, 64), cells_per_block=(8, 8), visualise=True)
        train_fv.append(feat)
        train_label.append(train_labels[i])

        cv2.imshow("HOG Img", hog_img)
        cv2.waitKey(1)

    print
    print "--------------------------------------------------------"
    rt = train_random_forest(np.asarray(train_fv, dtype='float32'), np.asarray(train_label, dtype='float32'))

    return rt


def compute_confusion(labels_gt, labels_test):
    """
        Spocita matici zamen.
        :param labels_test:
        :param labels_gt:
    """
    s = set(labels_gt)
    confusion = np.zeros((len(s), len(s)), np.int32)

    for i, j in zip(labels_gt, labels_test):
        confusion[i, j] += 1

    return confusion


def test(cls_path, n_cls, rf):
    """
        Nacte obrazky, spocita FV a klasifikuje na vypoctenem Random Forestu.
        :param cls_path:
        :param n_cls:
        :param rf:
    """
    print "--------------------------------------------------------"
    print "Vypocet FV z testovaci sady:"

    test_data, test_labels = read_data(cls_path, n_cls)
    test_fv = []
    test_label = []
    predicted = []

    for i, d in enumerate(test_data):
        print ".",
        img = bgr_dist(d)

        feat, hog_img = hog(img, orientations=8, pixels_per_cell=(64, 64), cells_per_block=(8, 8), visualise=True)
        test_fv.append(feat)
        test_label.append(test_labels[i])
        predicted.append(rf.predict(np.asarray(feat, dtype='float32')))

        cv2.imshow("HOG Img", hog_img)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    predicted = np.squeeze(predicted)
    print
    print "--------------------------------------------------------"
    print "Vysledky klasifikace:"
    print "Chybne klasifikovano:",
    print np.sum(predicted != np.asarray(test_label, dtype='float32')),
    print "/", len(predicted)
    err = (predicted != np.asarray(test_label, dtype='float32')).mean()
    print 'Error: %.2f %%' % (err * 100)

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

    trained_rf = train(clsTrainPath, n_class)
    test(clsTestPath, n_class, trained_rf)

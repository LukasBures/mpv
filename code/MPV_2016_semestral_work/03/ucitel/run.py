# -*- coding: utf-8 -*-
"""
Created on 19:29 12.11.15 

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits: 
@version: 1.0.0
"""

import mpv03_ucitel
import cv2
import numpy as np
from random import randint


def check_labels(test, gt):
    if len(test) is not len(gt):
        assert "Bad count of labels"

    good = 0
    for i in range(len(gt)):
        if test[i] == gt[i]:
            good += 1

    return good


def prepare_data(img, length):
    img = np.array(img)
    h, w = img.shape

    t_datas = []
    t_labels = []

    if w % length != 0 or h % length != 0:
        assert "Bad size of input image."

    w_bin = w / length
    h_bin = h / length

    i = 0
    for hh in range(h_bin):
        if hh % 5 == 0 and hh != 0:
            i += 1

        for ww in range(w_bin):
            h_from = hh * length
            h_to = (hh + 1) * length
            w_from = ww * length
            w_to = (ww + 1) * length
            sub_img = img[h_from:h_to, w_from:w_to]

            t_datas.append(sub_img)
            t_labels.append(i)

    return t_datas, t_labels


def select_data(datas, labels, count):
    t_datas = []
    t_labels = []

    for i in range(count):
        pos = randint(0, len(datas) - 1)
        t_datas.append(datas[pos])
        t_labels.append(labels[pos])

    return t_datas, t_labels


def compute_confusion(labels_gt, labels_test):
    s = set(labels_gt)
    confusion = np.zeros((len(s), len(s)), np.int32)

    for i, j in zip(labels_gt, labels_test):
        confusion[i, j] += 1

    return confusion


if __name__ == '__main__':
    test_img = cv2.imread("../data/test/Test.png", cv2.IMREAD_GRAYSCALE)
    train_img = cv2.imread("../data/train/Train.png", cv2.IMREAD_GRAYSCALE)
    digit_len = 20

    train_data_all, train_labels_all = prepare_data(train_img, digit_len)
    test_data_all, test_labels_gt_all = prepare_data(test_img, digit_len)

    test_data_random, test_labels_gt_random = select_data(test_data_all, test_labels_gt_all,
                                                          randint(len(test_data_all) / 2, len(test_data_all)))

    test_labels = mpv03_ucitel.classify(train_data_all, train_labels_all, test_data_random)

    ok = check_labels(test_labels, test_labels_gt_random)

    confusion_matrix = compute_confusion(test_labels_gt_random, test_labels)
    print "Confusion matrix:"
    print confusion_matrix
    print str(ok) + "/" + str(len(test_labels_gt_random)) + ", " + str(
        float(ok) / float(len(test_labels_gt_random)) * 100.0) + "%"

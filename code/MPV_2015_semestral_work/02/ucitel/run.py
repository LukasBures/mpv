# -*- coding: utf-8 -*-
"""
Created on 13:39 6.10.15 

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits: 
@version: 2.0.0
"""

import mpv02_ucitel
import cv2
import os


def prepare_data(data_path, n_class):
    datas = []
    labels = []

    for i in range(0, n_class):
        p = data_path + "/" + str(i) + "/"
        for fn in os.listdir(p):
            gs = cv2.imread(p + fn, cv2.CV_LOAD_IMAGE_GRAYSCALE)
            datas.append(gs)
            labels.append(i)

    return datas, labels


def check_labels(test, gt):
    if len(test) is not len(gt):
        assert "Bad count of labels"

    good = 0
    for i in range(len(gt)):
        if test[i] == gt[i]:
            good += 1

    return good


if __name__ == '__main__':
    n_visual_words = 10  # Pocet vizualnich slov v bow metode.
    train_data, train_label = prepare_data("../data/train/", 5)
    test_data, test_label_gt = prepare_data("../data/test/", 5)

    test_label = mpv02_ucitel.bow(train_data, train_label, test_data, n_visual_words)
    ok = check_labels(test_label, test_label_gt)

    print str(ok) + "/" + str(len(test_label_gt)) + " was classified ok."

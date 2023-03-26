# -*- coding: utf-8 -*-

import numpy as np
import cv2
from skimage.feature import hog

#nastaveni hogu
ori = 8
ppc = 4
cpb = 2

def countHog(data, labels):
    featureVec = list()
    label = list()

    for i in range(len(data)):
        img = data[i]
        feat, hog_img = hog(img, orientations=ori, pixels_per_cell=(ppc, ppc), cells_per_block=(cpb, cpb), visualise=True)
        featureVec.append(feat)
        label.append(labels[i])

    return featureVec, label


def train(imgs, labels):
    train_fv, train_label = countHog(imgs, labels)

    svm = cv2.ml.SVM_create()
    svm.train(np.asarray(train_fv, dtype='float32'), cv2.ml.ROW_SAMPLE, np.asarray([train_label], dtype='int32'))

    return svm


def test(svm, test_data):
    test_labels = np.zeros(len(test_data))

    #test_fv, test_label = countHog(test_data, test_labels)
    test_fv, test_label = countHog(test_data, test_labels)
    _, predicted = svm.predict(np.asarray(test_fv, dtype='float32'))

    result = list()

    for i in range(len(predicted)):
        result.append(predicted[i][0])

    return result


def classify(train_data, train_label, test_data):

    trained_svm = train(train_data, train_label)
    cls = test(trained_svm, test_data)

    return cls

# -*- coding: utf-8 -*-

import numpy as np
import cv2
from skimage.feature import hog

#nastaveni hogu
# s cislama bylo nutne experimentovat, aby byly hogy v obrazku spravne interpretovany
ori = 8
ppc = 4
cpb = 2

def countHog(data, labels):
    # spocita hogy v jednotlivych obrazcich a vrati feature vektor a labely
    featureVec = list()
    label = list()

    for i in range(len(data)):
        img = data[i]
        feature = hog(img, orientations=ori, pixels_per_cell=(ppc, ppc), cells_per_block=(cpb, cpb), visualise=True)
        featureVec.append(feature)
        label.append(labels[i])

    return featureVec, label


def train(imgs, labels):
    # spocte hogy na trenovaci sade a vytvori svm, ktery natrenuje a vrati
    train_fv, train_label = countHog(imgs, labels)

    svm = cv2.ml.SVM_create()
    svm.train(np.asarray(train_fv, dtype='float32'), cv2.ml.ROW_SAMPLE, np.asarray([train_label], dtype='int32'))

    return svm


def test(svm, test_data):
    # spocita hogy pro testovaci sadu a pomoci svm a funkce predict vyhodnoti natrenovane data s testovacimi
    test_labels = np.zeros(len(test_data))

    test_featureVec = countHog(test_data, test_labels)
    _, predicted = svm.predict(np.asarray(test_featureVec, dtype='float32'))

    result = list()

    for i in range(len(predicted)):
        result.append(predicted[i][0])

    return result


def classify(train_data, train_label, test_data):
    # spusti trenovani a testovani dat a vrati list, ktery udava poradi jednotlivych trid
    trained_svm = train(train_data, train_label)
    cls = test(trained_svm, test_data)

    return cls

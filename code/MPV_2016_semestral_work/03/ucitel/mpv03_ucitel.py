# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 12:58:47 2014

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits:
@version: 2.0.0
"""

import numpy as np
import cv2
from numpy.linalg import norm

SZ = 20
CLASS_N = 10


def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    m = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, m, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        b = np.int32(bin_n * ang / (2 * np.pi))
        bin_cells = b[:10, :10], b[10:, :10], b[:10, 10:], b[10:, 10:]
        mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:],  mag[10:, 10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)


class SVM:
    def __init__(self, c=2.67, gamma=5.383):
        self.model = cv2.ml.SVM_create()
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)
        self.model.setC(c)
        self.model.setGamma(gamma)

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        _, predicted = self.model.predict(samples)
        return predicted.ravel()


def evaluate_model(model, fv):
    labels = model.predict(fv)
    labels = labels.astype(dtype="int32")
    return labels


def classify(train_data, train_label, test_data):
    print "-------------------------------------------------------------------"
    print "TRAIN"
    print
    train_deskewed_data = map(deskew, train_data)
    train_fv = preprocess_hog(train_deskewed_data)

    print 'Training SVM ...'
    model = SVM()
    model.train(train_fv, np.array(train_label, dtype="int32"))

    print "-------------------------------------------------------------------"
    print "TEST"
    print
    test_deskewed_data = map(deskew, test_data)
    test_fv = preprocess_hog(test_deskewed_data)

    print 'Testing data ...'
    test_labels = evaluate_model(model, test_fv)
    print "-------------------------------------------------------------------"

    return test_labels

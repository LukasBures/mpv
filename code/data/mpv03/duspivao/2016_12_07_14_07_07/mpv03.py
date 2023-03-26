# -*- coding: utf-8 -*-
import numpy as np
import cv2

"""
@author: Bc. Ondrej Duspiva
@version: 1.0.0
"""

# ------------------------------------------------------------------------------
SZ = 20
affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
def transform(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=affine_flags)
    return img


def makeRowFromImg(img):
    w, h = np.size(img, 0), np.size(img, 0)
    result = list()
    for i in range(w):
        for j in range(h):
            result.append(img[i][j])
    result = np.asarray(result, dtype=np.float32)/max(result)
    return result

def classify(train_data, train_label, test_data):
    trainDataList = list()
    for i in range(len(train_data)):
        temp = transform(train_data[i])
        trainDataList.append(makeRowFromImg(temp))

    print
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.train(np.asarray(trainDataList, dtype='float32'), cv2.ml.ROW_SAMPLE, np.asarray([train_label], dtype='int32'))

    fvTest = list()
    for i in range(len(test_data)):
        fvTest.append(makeRowFromImg(train_label[i]))
    _, predicted = svm.predict(np.asarray(fvTest, dtype='float32'))
    predicted = np.squeeze(predicted)
    predicted = predicted.astype(dtype="int32")
    print predicted

    return predicted

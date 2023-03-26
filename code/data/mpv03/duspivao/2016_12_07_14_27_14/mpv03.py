# -*- coding: utf-8 -*-
import numpy as np
import cv2
from skimage.feature import hog


"""
@author: Bc. Ondrej Duspiva
@version: 1.0.0
"""

# ------------------------------------------------------------------------------
SZ = 20
affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
orient = 8
ppC1 = 5
ppC2 = 5

cpB1 = 2
cpB2 = 2

def transform(img):
    # metoda provede transformaci (narovnani obrazku)
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=affine_flags)
    return img


def makeRowFromImg(img):
    # spocteni relativniho histogramu. Nejdriv prevedeme dvojrozmerny obrazek na jednu radku, pak se jen jen podeli
    # tento vektor maximalni hodnotou, aby byl histogram normovany do 1
    w, h = np.size(img, 0), np.size(img, 0)
    result = list()
    for i in range(w):
        for j in range(h):
            result.append(img[i][j])
    result = np.asarray(result, dtype=np.float32)/max(result)
    return result

def classify(train_data, train_label, test_data):
    # pro kazdy z mnoziny trenovacich obrazku je preveden na vektor, to ale nebylo zrovna ucinne a tak se spoctou jeste
    # hogy, ty pak naskladame za vektor reprezentujici obrazek z cehoz vznika feature vector. Pomoci mnoziny techto vektoru
    # je pak natrnovano svm. Stejne se postupuje v pripade testovacich obrazku
    trainDataList = list()
    for i in range(len(train_data)):
        temp = transform(train_data[i])
        fvTemp = makeRowFromImg(temp)
        feat, _ = hog(temp, orientations=orient, pixels_per_cell=(ppC1, ppC2), cells_per_block=(cpB1, cpB2), visualise=True)
        fvFin = np.zeros((len(fvTemp)+len(feat)))
        for j in range(len(fvTemp)+len(feat)):
            if j < len(fvTemp):
                fvFin[j] = fvTemp[j]
        else:
            fvFin[j-len(fvTemp)] = fvFin[j-len(fvTemp)]

        trainDataList.append(fvFin)

    print "here"
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.train(np.asarray(trainDataList, dtype='float32'), cv2.ml.ROW_SAMPLE, np.asarray([train_label], dtype='int32'))
    print "here2"
    fvTest = list()
    for i in range(len(test_data)):
        temp = transform(test_data[i])
        fvTemp = makeRowFromImg(test_data[i])
        feat, _ = hog(temp, orientations=orient, pixels_per_cell=(ppC1, ppC2), cells_per_block=(cpB1, cpB2), visualise=True)
        fvFin = np.zeros((len(fvTemp)+len(feat)))
        for j in range(len(fvTemp)+len(feat)):
            if j < len(fvTemp):
                fvFin[j] = fvTemp[j]
        else:
            fvFin[j-len(fvTemp)] = fvFin[j-len(fvTemp)]
        fvTest.append(fvFin)
    _, predicted = svm.predict(np.asarray(fvTest, dtype='float32'))
    print "here3"
    predicted = np.squeeze(predicted)
    predicted = predicted.astype(dtype="int32")
    print predicted

    return predicted

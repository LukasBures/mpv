# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 13:38:32 2014



@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits:
@version: 2.0.0
"""
import numpy as np
import cv2
from skimage.morphology import skeletonize, binary_closing, dilation
from skimage.feature import hog


# ------------------------------------------------------------------------------
def normalize(img): # normalizace p?ed trénováním nebo klasifikováním
    deskewed = deskew(img)
    skelet = skeleton(deskewed)
    final = skelet
    # skelet = skeleton(img)
    # deskewed = deskew(skelet)
    # final = deskewed
    return final

def skeleton(img): # skeletonizace ?íslic po jejich 'srovnání
    img = np.array( img > 1)
    img = binary_closing(img)

    return skeletonize(img).astype(np.uint8)

def best_rot(list):
    # not used
    best = np.zeros(20)
    max = 0
    for i in list:
        np.sum(i, axis = 1)



def rotation(img, angle):
    # not used
    rows, cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def deskew(img): #Transformace sklonu písma, sou?ást normalizace
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*20*skew],[0, 1, 0]])
    img = cv2.warpAffine(img, M, (20, 20), flags = cv2.WARP_INVERSE_MAP + cv2.INTER_LINEAR)
    return img

def image(list): # vizualizace výsledk?
    img = np.zeros((1000,1400))
    number = 0
    stop = len(list)
    for i in range(50):
        for j in range(70):
            img[i * 20 : (i+1) * 20,j * 20 : (j+1) * 20 ] = list[number]
            number += 1
            if number == stop:
                break
        if number == stop:
            break
    cv2.imshow('img', img)
    cv2.waitKey(0)

def feature(img):
    feat = hog(img, orientations = 4, pixels_per_cell = (2, 2), cells_per_block = (10, 10))
    # feat = hog(img, orientations = 8, pixels_per_cell = (5, 5), cells_per_block = (4, 4))
    vector = np.asarray(feat , dtype='float32')
    return vector

def classify(train_data, train_label, test_data):
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

    # normalized = map(normalize, train_data)
    # featured = map(feature, normalized)

    featured = map(feature, train_data)

    featured = np.asarray(featured, dtype=np.float32)
    label = np.asarray(train_label, dtype=np.int32)

    classifier = cv2.ml.SVM_create()
    classifier.setType(cv2.ml.SVM_C_SVC)
    classifier.setKernel(cv2.ml.SVM_RBF)
    classifier.train(featured, cv2.ml.ROW_SAMPLE, label)

    cls = list()
    # normalized = map(normalize, test_data)
    # feat = map(feature, normalized)

    feat = map(feature, test_data)
    _ ,cls = classifier.predict(np.asarray(feat, dtype='float32'))
    cls = np.squeeze(cls)
    cls = cls.astype(dtype='int32')
    print np.sum(cls == np.asarray(train_label, dtype='float32'))/float(len(train_label))
    return cls


# img = cv2.imread('../Train.png', cv2.COLOR_BGR2GRAY)
# train_data = list()
# train_label = list()
# test_data = list()
# trida = 0
# for i in range(img.shape[0]/20):
#     for j in range(img.shape[1]/20):
#         train_data.append(img[i * 20 : (i+1) * 20,j * 20 : (j+1) * 20 ])
#         train_label.append(trida)
#     if (i+1)%5 == 0:
#         trida +=1
#
# test_data = train_data
# classify(train_data, train_label,test_data)
# #cv2.imshow('MNIST', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows

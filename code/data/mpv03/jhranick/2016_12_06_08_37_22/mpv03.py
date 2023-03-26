# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 13:38:32 2014

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits: 
@version: 2.0.0
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
from random import shuffle

# Logging utility initialization
logger = logging.getLogger('mpv02')
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

logger.addHandler(ch)


def split_images(img, step=20):
    """
    Split images into sub-images of given size (rectangular)
    :param img: Input image to be split
    :param step: Size of rectangular cut-off
    :return: List of split images
    """
    imgs = []

    (height, width) = img.shape
    for i in range(0, height, step):
        for j in range(0, width, step):
            slc = img[i:i+step, j:j+step]
            imgs.append(slc)

    return imgs


def train_svm(data, responses):
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.train(data, cv2.ml.ROW_SAMPLE, responses)

    return svm


def compute_confusion(labels_gt, labels_test):
    """
        Spocita matici zamen.
    """

    # TODO: implementace matice zamen
    s = set(labels_gt)
    confusion = np.zeros((len(s), len(s)), np.int32)

    for i, j in zip(labels_gt, labels_test):
        confusion[i, j] += 1

    return confusion


def get_features(img):
    fv = np.zeros(256)
    sift = cv2.xfeatures2d.SIFT_create(20)
    (kps, descs) = sift.detectAndCompute(img, None)

    if descs is None:
        descs = np.zeros(128)
    descs = descs.sum(0)
    fv[:128] = descs

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)

    sobel = sobelx * sobely
    normSobel = sobel + np.abs(sobel.min())
    normSobel = np.uint8(255 * (normSobel / normSobel.max()))

    ret, dst = cv2.threshold(normSobel, 0.75 * normSobel.max(), 255, 0)

    # plt.imshow(image, cmap='gray')
    # plt.show()
    #
    # dst = cv2.cornerHarris(image, 2, 3, 0.04)
    # #dst = cv2.dilate(dst, None)
    # ret, dst = cv2.threshold(dst, 0.25 * dst.max(), 255, 0)
    # dst = np.uint8(dst)
    # plt.imshow(dst, cmap='gray')
    # plt.show()
    # dst = np.where(dst >= 250)
    # print dst

    pts = np.where(dst == 255)
    pts = np.concatenate(pts).shape
    fv[128:] = pts[:128] if len(pts) > 128 else pts
    return fv


# ------------------------------------------------------------------------------
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
    # ----- TRAINING -----
    # TODO: Train data normalization

    # TODO: Get the features vector for each train image
    train_fv = []
    for i in range(len(train_data)):
        image = train_data[i]

        fv = get_features(image)
        train_fv.append(fv)

    logger.debug("Feature vectores obtained")

    train_fv = np.asarray(train_fv, dtype=np.float32)
    train_label = np.asarray([train_label], dtype=np.int32)
    svm = train_svm(train_fv, train_label)
    logger.debug("SVM trained")

    # ----- TESTING -----
    test_fv = []
    for i in range(len(test_data)):
        image = test_data[i]

        fv = get_features(image)
        test_fv.append(fv)

    _, predicted = svm.predict(np.asarray(test_fv, dtype=np.float32))
    predicted = np.squeeze(predicted)
    predicted = predicted.astype(dtype=np.int32)

    # print "--------------------------------------------------------"
    # print "Vysledky klasifikace:"
    # print "Chybne klasifikovano:",
    # print np.sum(predicted != np.asarray(testLabels, dtype='float32')),
    # print "/", len(predicted)
    # err = (predicted != np.asarray(testLabels, dtype='float32')).mean()
    # print 'Error: %.2f %%' % (err * 100)
    # print "--------------------------------------------------------"
    # print

    # confusion = compute_confusion(testLabels, list(predicted))
    # print "--------------------------------------------------------"
    # print "Confusion matrix:"
    # print confusion
    # print "--------------------------------------------------------"
    # print

    cls = predicted
    return cls


if __name__ == '__main__':
    # Load training image and split it to sub-images of 20x20 pxs
    imgTrainPath = r"D:/OneDrive/Faculty of Applied Science/Postgraduate Master Studies/2016-2017/MPV/" \
                   r"Semester Projects Repository\SP3\Train.png"
    imgTrain = cv2.imread(imgTrainPath, cv2.IMREAD_GRAYSCALE)
    logger.debug("Train image loaded in grayscale")
    imgs = split_images(imgTrain)

    # Create labels for sub-images
    imgsLabels = np.zeros(len(imgs), dtype=np.uint8)
    for i in range(10):
        imgsLabels[i * 350:i * 350 + 350] = np.full(350, i, dtype=np.uint8)
    logger.debug("Labels for sub-images created")

    # Generate randomly shuffle indices and create train and test data sets
    indices = np.arange(0, len(imgs), dtype=np.uint8)
    shuffle(indices)

    trainNum = int(len(imgs) * 0.9)
    trainData = []
    trainLabels = []
    testData = []
    testLabels = []
    for i in indices[:trainNum]:
        trainData.append(imgs[i])
        trainLabels.append(i)
    for i in indices[trainNum:]:
        testData.append(imgs[i])
        testLabels.append(i)
    logger.debug("Train and test data created")
    logger.debug("\nTrain data: %s\nTest data: %s" % (len(trainData), len(testData)))

    # Classify test data
    cls = classify(trainData, trainLabels, testData)
    # print(testLabels)
    # print(cls)



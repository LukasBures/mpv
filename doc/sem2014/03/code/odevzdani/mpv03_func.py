# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 12:58:47 2014

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits: 
@version: 1.0.0
"""

import numpy as np
import cv2
from numpy.linalg import norm

SZ = 20 # size of each digit is SZ x SZ
CLASS_N = 10

def Split2d(img, cell_size, flatten=True):
    h, w = img.shape[:2]
    sx, sy = cell_size
    cells = [np.hsplit(row, w//sx) for row in np.vsplit(img, h//sy)]
    cells = np.array(cells)
    if flatten:
        cells = cells.reshape(-1, sy, sx)
    return cells

def LoadDigits(fileName):
    print 'loading "%s" ...' % fileName
    digits_img = cv2.imread(fileName, 0)
    digits = Split2d(digits_img, (SZ, SZ))
    labels = np.repeat(np.arange(CLASS_N), len(digits) / CLASS_N)
    return digits, labels

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img
    
def PreprocessHog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n * ang / (2 * np.pi))
        bin_cells = bin[:10, :10], bin[10:, :10], bin[:10, 10:],  bin[10:, 10:]
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

class StatModel(object):
    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)
        
class SVM(StatModel):
    def __init__(self, C = 1, gamma = 0.5):
        self.params = dict( kernel_type = cv2.SVM_RBF,
                            svm_type = cv2.SVM_C_SVC,
                            C = C,
                            gamma = gamma )
        self.model = cv2.SVM()

    def train(self, samples, responses):
        self.model = cv2.SVM()
        self.model.train(samples, responses, params = self.params)

    def predict(self, samples):
        return self.model.predict_all(samples).ravel()

def EvaluateModel(model, digits, samples, labels):
    resp = model.predict(samples)
    err = (labels != resp).mean()
    print "Chybne klasifikovano:", np.sum(labels != resp), "/", len(labels), ', error: %.2f %%' % (err*100)
    print
    
    confusion = np.zeros((10, 10), np.int32)
    for i, j in zip(labels, resp):
        confusion[i, j] += 1
    print 'confusion matrix:'
    print confusion
    print

def SVMtest(Imgs):
    print __doc__
    TRAIN_FN = 'Train.png'
    #TEST_FN = 'Test.png'

    print
    print "-------------------------------------------------------------------"
    print "TRAIN"
    print
    trainDigits, trainLabels = LoadDigits(TRAIN_FN)
    trainDeskewDigits = map(deskew, trainDigits)
    trainFVhog = PreprocessHog(trainDeskewDigits)

    print 'Training SVM ...'
    model = SVM(C = 2.67, gamma = 5.383)
    model.train(trainFVhog, trainLabels)
    
    print 'Saving SVM as "digits_svm.dat"...'
    model.save("DigitsSVM.dat")


    print
    print "-------------------------------------------------------------------"
    print "TEST"
    print
    #testDigits, testLabels = LoadDigits(TEST_FN)
    #testDeskewDigits = map(deskew, testDigits)
    testDeskewDigits = map(deskew, Imgs)
    testFVhog = PreprocessHog(testDeskewDigits)

    print 'Testing SVM ...'
    resp = model.predict(testFVhog)
    #EvaluateModel(model, testDigits, testFVhog, testLabels)
    model.save("DigitsSVM.dat")
    cls = np.zeros((len(Imgs), 1), np.int)
    cls[:, 0] = resp[:] 
    return cls
    











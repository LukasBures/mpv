# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 15:47:41 2014

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits: 
@version: 1.0.0
"""

import cv2
import numpy as np
import os, glob, sys
from skimage.feature import hog


#------------------------------------------------------------------------------   
def BGRdist(Img):
    ImgOUT = np.sqrt(np.power(Img[:, :, 0], 2.0) + np.power(Img[:, :, 1], 2.0)
    + np.power(Img[:, :, 2], 2.0))
    ImgOUT = (ImgOUT / ImgOUT.max()) * 255
    ImgOUT = np.uint8(ImgOUT)
    return ImgOUT

#------------------------------------------------------------------------------    
def TrainSVM(data, responses):
    svmParams = dict(kernel_type = cv2.SVM_LINEAR, svm_type = cv2.SVM_C_SVC)
    svm = cv2.SVM()
    print "--------------------------------------------------------"
    print "Trenovani SVM"
    svm.train_auto(data, responses, None, None, svmParams)
    print "--------------------------------------------------------"
    return svm
    
#------------------------------------------------------------------------------    
if __name__ == "__main__":
    clsPath = ["drevoTapeta", "kava", "kosik", "zed", "zelezo"]
    clsTrainPath = "/ImgTex/train/"
    clsTestPath = "/ImgTex/test/"
    cv2.namedWindow("HOG Img", 0)




    #-TRAIN--------------------------------------------------------------------
    print    
    print "--------------------------------------------------------"
    print "Vypocet FV z trenovaci sady:"
    trainData = list()
    trainResponses = list()

    folder = 0    

    for i in range(len(clsPath)):
        path = os.path.dirname(sys.argv[0]) + clsTrainPath + clsPath[folder]
        os.chdir(path)
        print path 
        
        for file in glob.glob("*.jpg"):
            
            Img = cv2.imread(file)    
            Img = BGRdist(Img)
            feat, hogImg = hog(Img, orientations = 8, pixels_per_cell = (64, 64),
                               cells_per_block = (8, 8), visualise = True)            
            trainData.append(feat)
            trainResponses.append(i)
    
            cv2.imshow("HOG Img", hogImg)
            cv2.waitKey(1)
            
        folder += 1 
        
    print "--------------------------------------------------------"
    print
    svm = TrainSVM(np.asarray(trainData, dtype = 'float32'),
                   np.asarray(trainResponses, dtype = 'float32'))




    #-TEST--------------------------------------------------------------------
    print    
    print "--------------------------------------------------------"
    print "Vypocet FV z testovaci sady:"
    testData = list()
    testResponses = list()
    folder = 0    

    for i in range(len(clsPath)):
        path = os.path.dirname(sys.argv[0]) + clsTestPath + clsPath[folder]
        os.chdir(path)
        print path 
        
        for file in glob.glob("*.jpg"):
            
            Img = cv2.imread(file)    
            Img = BGRdist(Img)
            feat, hogImg = hog(Img, orientations = 8, pixels_per_cell = (64, 64),
                               cells_per_block = (8, 8), visualise = True)            
            testData.append(feat)
            testResponses.append(i)
    
            cv2.imshow("HOG Img", hogImg)
            cv2.waitKey(1)
            
        folder += 1 
        
    cv2.destroyAllWindows()
    print "--------------------------------------------------------"    
    print
    predicted = svm.predict_all(np.asarray(testData, dtype = 'float32'))        
    predicted = np.squeeze(predicted)
    
    print "--------------------------------------------------------"    
    print "Vysledky klasifikace:"
    print "Chybne klasifikovano:",
    print np.sum(predicted != np.asarray(testResponses,dtype = 'float32')),
    print "/", len(predicted)
    err = (predicted != np.asarray(testResponses, dtype = 'float32')).mean()
    print 'Error: %.2f %%' % (err * 100)
    print "--------------------------------------------------------"    
    print 
        
    confusion = np.zeros((5, 5), np.int32)
    for i, j in zip(np.int32(testResponses), np.int32(predicted)):
        confusion[i, j] += 1
    print "--------------------------------------------------------"    
    print "Confusion matrix:"
    print confusion
    print "--------------------------------------------------------"    
    print
    
    
    
    
    
    
    
    
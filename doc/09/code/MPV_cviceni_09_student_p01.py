# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 10:45:16 2014

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





    return ImgOUT

#------------------------------------------------------------------------------    
def TrainRandomForest(data, responses):







    return rF
    
#------------------------------------------------------------------------------    
if __name__ == "__main__":
    clsPath = ["drevoTapeta", "kava", "kosik", "zed", "zelezo"]
    clsTrainPath = "\\ImgTex\\train\\"
    clsTestPath = "\\ImgTex\\test\\"
    cv2.namedWindow("HOG Img", 0)




    #-TRAIN--------------------------------------------------------------------
    print    
    print "--------------------------------------------------------"
    print "Vypocet FV z trenovaci sady:"




























    #-TEST--------------------------------------------------------------------
    print    
    print "--------------------------------------------------------"
    print "Vypocet FV z testovaci sady:"


































    
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

    
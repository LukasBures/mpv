# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 13:38:32 2014

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits: 
@version: 1.0.0
"""

import numpy as np
import cv2

#------------------------------------------------------------------------------
def CalculateSiftDescriptors(paths, nImg):
    sift = cv2.SIFT(125)    
    Descriptors = []
    nDescriptorsInClass = np.zeros((1, len(paths)), np.int)
    
    for i in range(len(paths)):
        for j in range(1, nImg + 1):
            if j < 10:
                name = paths[i] + "000" + str(j) + ".jpg"
            elif j < 100:
                name = paths[i] + "00" + str(j) + ".jpg"
            else:
                name = paths[i] + "0" + str(j) + ".jpg"
            
            print name
          
            ImgGS = cv2.imread(name , cv2.CV_LOAD_IMAGE_GRAYSCALE)    
            
            _, des = sift.detectAndCompute(ImgGS, None)   
            
            if i == 0 and j == 1:
                Descriptors = des
            else:            
                Descriptors = np.concatenate([Descriptors, des])   
                
            nDescriptorsInClass[0, i] = nDescriptorsInClass[0, i] + des.shape[0]
                
    return Descriptors, nDescriptorsInClass

#------------------------------------------------------------------------------
def CalculateCentroids(Descriptors, nVisualWord):
    # Flag to specify the number of times the algorithm is executed using
    # different initial labellings. The algorithm returns the labels that
    # yield the best compactness
    attempts = 10
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, Label, Center = cv2.kmeans(Descriptors, nVisualWord, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    
    return Label, Center

#------------------------------------------------------------------------------
def CalculateModels(Label, nDescriptorsInClass, nVisualWord):
    
    ModelHistograms = np.zeros((nDescriptorsInClass.shape[1], nVisualWord), np.float64)

    temp_nDescriptorsInClass = np.zeros((1, nDescriptorsInClass.shape[1] + 1), np.int)
    for i in range(nDescriptorsInClass.shape[1] + 1):
        if(i == 0):
            temp_nDescriptorsInClass[0, 0] = 0
        else:
            temp_nDescriptorsInClass[0, i] = temp_nDescriptorsInClass[0, i - 1] + nDescriptorsInClass[0, i - 1]
    
    for i in range(nDescriptorsInClass.shape[1]):
        for j in range(temp_nDescriptorsInClass[0, i], temp_nDescriptorsInClass[0, i + 1]):
            ModelHistograms[i, Label[j]] += 1.0
    
    ModelHistograms = np.float64(ModelHistograms) / np.float64(nDescriptorsInClass.T)
    
    return ModelHistograms


#------------------------------------------------------------------------------
sift = cv2.SIFT(125)        
def CalculateExampleHistogram(exmplPath, Centers):
    ImgGS = cv2.imread(exmplPath, cv2.CV_LOAD_IMAGE_GRAYSCALE)    
    _, des = sift.detectAndCompute(ImgGS, None)   
    
    des = np.float64(des)
    Label = np.zeros((1, des.shape[0]), np.int)    
    
    # najde nejblizsi tridu
    for i in range(des.shape[0]):
        distance = np.zeros((1, Centers.shape[0]), np.float64)
        for j in range(Centers.shape[0]):
            distance[0, j] = np.sqrt(np.sum(np.power(des[i] - Centers[j], 2.0)))
        mi = np.min(distance)
        
        for k in range(Centers.shape[0]):
            if mi == distance[0, k]:
                Label[0, i] = k
                break
    
    # vypocte histogram
    exampleHistogram = np.zeros((1, Centers.shape[0]), np.float64)
    for i in range(Label.shape[1]):
        exampleHistogram[0, Label[0, i]] += 1.0
    
    return exampleHistogram / des.shape[0]

#------------------------------------------------------------------------------
def CalculateAngle(ModelHistograms, exampleHist):
    
    ModelHistograms = np.float64(ModelHistograms)
    exampleHist = np.float64(exampleHist)
    
    res = np.zeros((1, ModelHistograms.shape[0]), np.float64)
    for i in range(ModelHistograms.shape[0]):
        a = np.sum(ModelHistograms[i, :] * exampleHist)
        b = np.sqrt(np.sum(np.power(ModelHistograms, 2)))
        c = np.sqrt(np.sum(np.power(exampleHist, 2)))
        res[0, i] = np.arccos( a / (b * c))
    
    mi = np.min(res)
    cls = []
    for i in range(res.shape[1]):
        if mi == res[0, i]:
            cls = i
            break

    return cls


#------------------------------------------------------------------------------
if __name__ == '__main__':
    paths = ["./ImgTex/drevoTapeta/", "./ImgTex/kava/", "./ImgTex/kosik/", "./ImgTex/zed/", "./ImgTex/zelezo/"]
    nImg = 60
    nVisualWord = 50
    
    Descriptors, nDescriptorsInClass = CalculateSiftDescriptors(paths, nImg)
    Label, Centers = CalculateCentroids(Descriptors, nVisualWord)
    ModelHistograms = CalculateModels(Label, nDescriptorsInClass, nVisualWord)
    
    nTestImg = 40
    pathsTest = ["./ImgTex/drevoTapeta/", "./ImgTex/kava/", "./ImgTex/kosik/", "./ImgTex/zed/", "./ImgTex/zelezo/"]
    good = np.zeros((1, len(pathsTest)), np.int)
    bad = np.zeros((1, len(pathsTest)), np.int)
    for i in range(len(pathsTest)):
        for j in range(nImg + 1, nImg + nTestImg + 1):

            if j < 10:
                exmplPath = pathsTest[i] + "000" + str(j) + ".jpg"
            elif j < 100:
                exmplPath = pathsTest[i] + "00" + str(j) + ".jpg"
            else:
                exmplPath = pathsTest[i] + "0" + str(j) + ".jpg"
                
                
            print exmplPath
            
            exampleHist = CalculateExampleHistogram(exmplPath, Centers)
            cls = CalculateAngle(ModelHistograms, exampleHist)
            
            if(i == cls):
                good[0, i] += 1
            else:
                bad[0, i] += 1

    print "-------------------------------------------------------------------"
    print "GOOD:", good
    print "BAD: ", bad
    print "Uspesnost:", np.float64(good) / (np.float64(good) + np.float64(bad)) 
    print "-------------------------------------------------------------------"

                
                
                
                
                
                
                

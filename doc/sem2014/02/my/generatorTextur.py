# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 13:36:41 2014

@author: Lukas Bures
"""

import numpy as np
import cv2



if __name__ == '__main__':
    
    imgRead = ["./ImgTex/drevoTapeta.png", "./ImgTex/kava.jpg", 
               "./ImgTex/kosik.jpg", "./ImgTex/zed.jpg",
               "./ImgTex/zelezo.jpg"]
    outFolder = ["./ImgTex/drevoTapeta/", "./ImgTex/kava/", 
               "./ImgTex/kosik/", "./ImgTex/zed/",
               "./ImgTex/zelezo/"]
               
    nImg = 100
    sz = 512
    for i in range(len(imgRead)):
        path = imgRead[i]
        print path
        Img = cv2.imread(path, cv2.CV_LOAD_IMAGE_COLOR)
        xLim = Img.shape[1] - sz
        yLim = Img.shape[0] - sz
        
        for j in range(1, nImg + 1):
            startX = np.random.randint(0, xLim)
            startY = np.random.randint(0, yLim)
            
            smallImg = Img[startY:startY + sz, startX: startX + sz]
            if(j < 10):
                pathToSmallImg = outFolder[i] + "000" + str(j) + ".jpg"
            elif(j < 100):
                pathToSmallImg = outFolder[i] + "00" + str(j) + ".jpg"
            else:
                pathToSmallImg = outFolder[i] + "0" + str(j) + ".jpg"
                
            cv2.imwrite(pathToSmallImg, smallImg)
            
            
            
            
            
            
            
            
            
            
            
        
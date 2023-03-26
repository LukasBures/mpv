# -*- coding: utf-8 -*-
"""
Created on Wed Oct 08 12:59:17 2014

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@version: 1.0.0
"""


# Import modulu.
import cv2 
import numpy as np 


#------------------------------------------------------------------------------
# Generator
#------------------------------------------------------------------------------


def harris(ImgGS, th):
    # Vypocet Harris Corner Detectoru.
    ImgGS = np.float32(ImgGS)
    blockSize = 3;
    apertureSize = 3;
    k = 0.04;
    HarrisImg = cv2.cornerHarris(ImgGS, blockSize, apertureSize, k);
        
    # Normalizace        
    NormHarris = HarrisImg + np.abs(HarrisImg.min())
    NormHarris = 255 * (NormHarris / NormHarris.max())
    
    cv2.imshow("pppppppppppppppppppppp", NormHarris)    
    cv2.waitKey(1)
    
    _, ImgTh = cv2.threshold(NormHarris, th, 255, cv2.THRESH_BINARY)
    
    
    cv2.imshow("ImgThImgThImgThImgThImgTh", ImgTh)    
    cv2.waitKey(1)
    
    
    
    return ImgTh
    
    
if __name__ == '__main__':
    
    nImg = 100
    srcFolderName = "./src/"    
    dstFolderName = "./dst/"    
    textFile = open("thValue.txt", "w")
    inExt = ".png"
    outExt = ".jpg"
    
    for i in range(nImg):
        if i < 10:
            name = "00" + str(i)
        elif i < 100:
            name = "0" + str(i)
        elif i < 1000:
            name = str(i)
            
        ImgGS = cv2.imread(srcFolderName + name + inExt, cv2.IMREAD_GRAYSCALE)
        th = np.random.randint(100, 150)
        
        ImgTh = harris(ImgGS, th)
        
        textFile.write(str(th) + "\r\n")
        cv2.imwrite(dstFolderName + name + outExt, ImgTh)
        print i, ": th =", th        
        
    textFile.close()
    
    
    
    
        
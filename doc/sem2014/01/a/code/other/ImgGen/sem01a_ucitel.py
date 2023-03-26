# -*- coding: utf-8 -*-
"""
Created on Wed Oct 01 17:13:56 2014

@author: Lukas Bures
"""

import numpy as np
import cv2
# from matplotlib import pyplot as plt 

sz = 512

def otsu(hist, total):
    """
    http://en.wikipedia.org/wiki/Otsu's_method
    """
    s = 0 # sum
    for i in range(256):
        s += i * hist[i]
        
    sB = 0.0 # 2nd sum
    wB = 0.0
    wF = 0.0
    ma = 0.0 # max
    between = 0.0
    th1 = 0.0
    th2 = 0.0
    
    for i in range(256):
        wB += hist[i]
        
        if(wB == 0):
            continue
        
        wF = total - wB
        
        if(wF == 0):
            break
        
        sB += i * hist[i] 
        mB = sB / wB
        mF = (s - sB) / wF
        between = wB * wF * np.power(mB - mF, 2)
        
        if(between >= ma):
            th1 = i
            if(between > ma):
                th2 = i
            ma = between

    return (th1 + th2) / 2.0
    
    

if __name__ == '__main__':

    textFile = open("thValue.txt", "w")
    nImgToGen = 100     
    
    for i in range(nImgToGen + 1):
        ImgOnes = np.ones((sz, sz), np.float)
        
        mean1 = np.random.randint(5, 45) / 100.0
        std1 = np.random.randint(5, 95) / 1000.0
        noise1 = np.abs(np.random.normal(mean1, std1, (sz, sz)))
        Img = np.subtract(ImgOnes, noise1)
        
        mean2 = np.random.randint(55, 95) / 100.0
        std2 = np.random.randint(5, 95) / 1000.0
        
        mode = np.random.randint(1, 4)
        if mode == 1:
            noise2 = np.abs(np.random.normal(mean2, std2, ((sz / 2) - 1, (sz / 2) - 1)))
            Img[1:(sz / 2), 1:(sz / 2)] = np.subtract(ImgOnes[1:(sz / 2), 1:(sz / 2)], noise2)
        elif mode == 2:
            noise2 = np.abs(np.random.normal(mean2, std2, (sz, (sz / 2) - 1)))
            Img[:, 1:(sz / 2)] = np.subtract(ImgOnes[:, 1:(sz / 2)], noise2)
        else:
            noise2 = np.abs(np.random.normal(mean2, std2, (sz, sz)))
            Img[:, 1:(sz / 2)] = np.subtract(ImgOnes[:, 1:(sz / 2)], noise2[:, 1:(sz / 2)])
            Img[1:(sz / 2), (sz / 2):sz] = np.subtract(ImgOnes[1:(sz / 2), (sz / 2):sz], noise2[1:(sz / 2), (sz / 2):sz])
        
        Img = np.uint8(255.0 * Img)

        # Ulozeni obrazku        
        path = "./ImgGen/syntheticImg_" + str(i) + ".jpg"
        print "Generating image number", i, "|", path,
    
        thresh, im_bw = cv2.threshold(Img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        print "|thresh", thresh, "|mode", mode, "|"
    
        #    plt.figure("Histogram")
        #    plt.hist(Img.flatten(), 256, [0, 256], color = 'r')
        #    plt.xlim([1, 256])
        #    plt.show()

        # hist, bins = np.histogram(Img.flatten(), 256, [0, 256])
        # th = otsu(hist, sz * sz)
        # print "manualy calculated =", th

        textFile.write(str(thresh) + "\r\n")
        cv2.imwrite(path, Img)
        
    textFile.close()
        
        
        
        
    
    
    
    
    

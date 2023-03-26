# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 21:51:30 2014

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits: 
@version: 1.0.0
"""

import cv2
import numpy as np

#------------------------------------------------------------------------------
# Priklad 2: Background Subtraction OpenCV implementace.
#------------------------------------------------------------------------------
def BS(bg, ImgGS):
    unit = 24
    th = 0.2
    fg = np.zeros((ImgGS.shape[0], ImgGS.shape[1]), np.float32)
    
    for i in range(ImgGS.shape[0] / unit):
        for j in range(ImgGS.shape[1] / unit):
            bg_hist, _ = np.histogram(bg[(i * unit):((i + 1) * unit), (j * unit):((j + 1) * unit)], 32, None, True)
            ImgGS_hist, _ = np.histogram(ImgGS[(i * unit):((i + 1) * unit), (j * unit):((j + 1) * unit)], 32, None, True)
            #val = cv2.compareHist(np.float32(bg_hist), np.float32(ImgGS_hist), 0)
            
            
                    
            [r, c] = LBPcode.shape
            LBPcode = LBPcode / 16
            FV = FV / 16
            distance = np.zeros(r, np.float64)        
        
        #---------------------------------------------------------------------------
        # Euklidovska vzdalenost.            
        #    for i in range(r):
        #        for j in range(c):
        #            distance[i] = distance[i] + (np.power(FV[j], 2) - np.power(LBPcode[i, j], 2))
        #        distance[i] = np.sqrt(distance[i])
        #    
        #---------------------------------------------------------------------------
        # Chi-square.
            for i in range(r):
                for j in range(c):
                    if FV[j] - LBPcode[i, j] == 0 or FV[j] == 0:
                        continue
                    else:
                        distance[i] = distance[i] + np.power((FV[j] - LBPcode[i, j]) , 2) / FV[j]
        #
        #---------------------------------------------------------------------------
        # Histogram intersection.
        #    for i in range(r):
        #        for j in range(c):
        #            if FV[j] < LBPcode[i, j]:
        #                distance[i] = distance[i] + FV[j]
        #            else:
        #                distance[i] = distance[i] + LBPcode[i, j]
        #            
        #    distance = 1 - distance
        #    
        #---------------------------------------------------------------------------
        # 
        #    for i in range(r):
        #        for j in range(c):
        #            if LBPcode[i, j] == 0 or FV[j] == 0:
        #                continue
        #            distance[i] = distance[i] + LBPcode[i, j] * np.log(LBPcode[i, j] / FV[j])
        #    
        #---------------------------------------------------------------------------

            if(np.abs(val) < th):
                fg[(i * unit):((i + 1) * unit), (j * unit):((j + 1) * unit)] = 1.0

    return fg

if __name__ == '__main__':
  
    cap = cv2.VideoCapture("Megamind.avi")
    cv2.namedWindow("Original Video", 1)
    cv2.namedWindow("Foreground Mask", 1)
    
    
    ret, Img = cap.read()
    if(ret):
        ImgGS = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)
        bg = ImgGS
        
        
    while(True):
        ret, Img = cap.read()       
        
        if(ret):
            ImgGS = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)
            
            cv2.imshow("Original Video", ImgGS)
            
            
            fg = BS(bg, ImgGS)
            
            cv2.imshow("Foreground Mask", fg)
            cv2.imshow("Background Mask", bg)
            bg = ImgGS
        if(cv2.waitKey(1) >= 0):
            break
    
    cap.release()
    cv2.destroyAllWindows()
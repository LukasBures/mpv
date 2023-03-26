# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 20:02:10 2014

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits: 
@version: 1.0.0
"""

import cv2 # Import OpenCV.
import numpy as np
import csv


def LBP(ImgGS):
    
    h, w = ImgGS.shape 
    
    if h == w == 512:
        
        szCell = 128
        xCell = w / szCell
        yCell = h / szCell
        Histograms = []
                
        
        for i in range(yCell):
            for j in range(xCell):
                
                Nums = []              
                    
                for k in range(1, szCell - 1):
                    for l in range(1, szCell - 1):
                        Y = (i * szCell) + k
                        X = (j * szCell) + l
                        
                        code = np.zeros((1, 8), np.int)
                         
                        code[0, 0] = 128 if ImgGS[Y, X] > ImgGS[Y - 1, X - 1] else 0
                        code[0, 1] = 64  if ImgGS[Y, X] > ImgGS[Y - 1, X    ] else 0
                        code[0, 2] = 32  if ImgGS[Y, X] > ImgGS[Y - 1, X + 1] else 0
                        code[0, 3] = 16  if ImgGS[Y, X] > ImgGS[Y    , X + 1] else 0
                        code[0, 4] = 8   if ImgGS[Y, X] > ImgGS[Y + 1, X + 1] else 0
                        code[0, 5] = 4   if ImgGS[Y, X] > ImgGS[Y + 1, X    ] else 0
                        code[0, 6] = 2   if ImgGS[Y, X] > ImgGS[Y + 1, X - 1] else 0
                        code[0, 7] = 1   if ImgGS[Y, X] > ImgGS[Y    , X - 1] else 0
                        
                        Nums.append(np.sum(code))
                
                hist, _ = np.histogram(Nums, 256, None, True, None, None)
                Histograms = np.concatenate([Histograms, hist])
                
        
        return Histograms
    else:
        print "Obrazek ma spatnou velikost:", w, "x", h
        return -1


#------------------------------------------------------------------------------
# Priklad 1: Local binary patterns (LBP)
#------------------------------------------------------------------------------
if __name__ == '__main__':

    nImg = 40
    with open('LBP.csv', 'wb') as csvfile:
        File = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        
        for i in range(nImg):
            if i < 10:
                path = "./textures/0" + str(i) + ".png"
            else:    
                path = "./textures/" + str(i) + ".png"
    
            print path
            
            # Nacteni barevneho obrazku.
            Img = cv2.imread(path, cv2.IMREAD_COLOR)
            
            # Prevede barevny obrazek do sedotonoveho formatu.
            ImgGS = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)
            
            FV = LBP(ImgGS)
    
            File.writerow(FV)
 
   




























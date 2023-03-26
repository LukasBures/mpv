# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 15:35:41 2014

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits: Ing. Marek Hruz, Ph.D., Ing. Petr Neduchal
@version: 1.0.0
"""

import cv2 # Import OpenCV.
import numpy as np
import csv
import time

#-----------------------------------------------------------------------------
# Vypocte LBP kody, histogramy a FV.
#-----------------------------------------------------------------------------
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
        
        return np.float64(Histograms)
    else:
        print "Obrazek ma spatnou velikost:", w, "x", h
        return -1

#-----------------------------------------------------------------------------
# Vrati indexy peti nejpodobnejsich textur.
#-----------------------------------------------------------------------------
def Find5BestMatch(FV, LBPcode):
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

    srt = np.sort(distance)
    idx = np.zeros(5, np.int)    
    for i in range(5):
        for j in range(len(distance)):
            if srt[i] == distance[j]:
                idx[i] = j
    return idx

#------------------------------------------------------------------------------
# Priklad 1: Local binary patterns (LBP)
#------------------------------------------------------------------------------
if __name__ == '__main__':
    start = time.time()
    nImg = 40
    
    with open('LBP.csv', 'rb') as csvfile:
        File = csv.reader(csvfile, delimiter=';', quotechar='|')
        r = 0
        LBPcode = np.zeros((nImg, 4096), np.float64)
        for row in File:
            LBPcode[r, :] = np.float64(row)
            r = r + 1
    
    # Databaze obrazku 0-39    .
    nImg = np.random.randint(0, nImg)
    if nImg < 10:
        path = "./textures/0" + str(nImg) + ".png"
    else:    
        path = "./textures/" + str(nImg) + ".png"

    # Nacte nahodny baervny obrazek a prevede ho do odstinu sedi.
    Img = cv2.imread(path, cv2.IMREAD_COLOR)
    ImgGS = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)
    
    # Zobrazi aktualne vybrany obrazek.
    cv2.namedWindow("Loaded image")
    cv2.imshow("Loaded image", Img)
    cv2.waitKey(1)
    
    # Spocita Feature Vector, zretezene histogramz LBP kodu.
    FV = LBP(ImgGS)

    if len(FV) != 1:
        # Vypocita nejblizsiho souseda pro 5 nejpodobnejsich textur.
        idx = Find5BestMatch(FV, LBPcode)
        
        # Vykresli 5 nejpodobnejsich textur.
        for i in range(len(idx)):
            if idx[i] < 10:
                path = "./textures/0" + str(idx[i]) + ".png"
            else:    
                path = "./textures/" + str(idx[i]) + ".png"
        
            Img = cv2.imread(path, cv2.IMREAD_COLOR)
            
            cv2.namedWindow(str(i) + " - " + path)
            cv2.imshow(str(i) + " - " + path, Img)
            cv2.waitKey(1)
    else:
         print "LBP nebyl spravne vypocten!"   
            
    end = time.time()
    print "Doba behu programu:", end - start, "[s]"
    cv2.waitKey()
    cv2.destroyAllWindows()
























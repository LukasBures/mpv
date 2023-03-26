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

#------------------------------------------------------------------------------
# Vypocte LBP kody, histogramy a FV.
#------------------------------------------------------------------------------
def LBP(ImgGS):
    
    h, w = ImgGS.shape 
    
    if h == w == 512:
   









        #----------------------------------------------------------------------
        # Vase implementace LBP.
        #----------------------------------------------------------------------
        Histograms = np.zeros((1, 4096))














        # Vrati vektor Histograms (16 zretezenych znormovanych histogramu).
        return np.float64(Histograms)
    else:
        print "Obrazek ma spatnou velikost:", w, "x", h
        return -1


#------------------------------------------------------------------------------
# Vrati indexy peti nejpodobnejsich textur.
#------------------------------------------------------------------------------
def Find5BestMatch(FV, LBPcode):
#------------------------------------------------------------------------------
# Normalizace FV a LBPcode (obsahuji 16 zretezenych znormovanych histogramu).

#------------------------------------------------------------------------------
# Euklidovska vzdalenost.            

#------------------------------------------------------------------------------
# Chi-square.

#------------------------------------------------------------------------------
# Histogram intersection.

#------------------------------------------------------------------------------
# 4

#------------------------------------------------------------------------------
# Vrati idx vektor s indexy peti nejpodobnejsimi texturami.
    idx = np.zeros(5, np.int)    
    
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
    
    # Databaze obrazku 0-39.
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
























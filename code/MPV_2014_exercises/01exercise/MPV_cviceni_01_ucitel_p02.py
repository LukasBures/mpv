# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 16:21:27 2014

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits: Ing. Marek Hruz, Ph.D.
@version: 1.0.0
"""


# Import OpenCV modulu.
import cv2 

# Import modulu pro pocitani s Pythonem.
import numpy as np 

# Import modulu pro vykreslovani grafu.
from matplotlib import pyplot as plt 


def NonMaximumSuppression(hist, halfWnd):
    """
    Funkce pro nalezeni maxim.
    """
    
    # Deklarace promennych.
    l = len(hist)
    out = np.zeros(len(hist), np.int8)
    maxInd = 0        
    i = 0
    
    # Hlavni cyklus.
    while i < l:
        if maxInd < (i - halfWnd):   
            maxInd = i - halfWnd

        e = min(i + halfWnd, (l - 1))
        
        while maxInd <= e:
            if hist[maxInd] > hist[i]:
                break
            
            maxInd = maxInd + 1
            
        if maxInd > e:
            out[i] = 1
            maxInd = i + 1
            i = i + halfWnd
            
        i = i + 1
    
    # Navrat pole s oznacenim pozic nalezenych maxim.
    # 1 na pozicich, kde bylo nalezeno maximum, jinak 0.
    return out


def main():
    """
    Hlavni funkce volana po spusteni programu.    
    """    
    
    # Nacteni sedotonoveho obrazku.
    ImgGS = cv2.imread("tsukuba_r.png", cv2.IMREAD_GRAYSCALE)
    
    # Vypocita histogram z obrazku.
    hist, bins = np.histogram(ImgGS.flatten(), 256, [0, 256])
    
    # Volba velikosti poloviny okenka, cele okenko = (2 * halfWnd) + 1.
    halfWnd = 50
    
    # Volani funkce, ktera nalezne maxima v histogramu.
    # Vrati pole s nenulovymi hodnotami na pozicich, kde bylo
    # nalezeno maximum.
    maxPositions = NonMaximumSuppression(hist, halfWnd)

    # Vytvori novou figuru, stejne jako v MATLABu.
    plt.figure("Histogram")
    
    # Vykresleni sloupcu histogramu cervenou barvou.
    plt.hist(ImgGS.flatten(), 256, [0, 256], color = 'r')

    # Prekresleni sloupcu histragramu, ve kterych bylo vyhodnoceno 
    # maximum, modrou barvou.
    for i in range(len(maxPositions)):
        if maxPositions[i] != 0:
            plt.bar(i, hist[i], width = 1, color = 'b', hold = None)
    
    # Nastaveni limitu X-ove osy.
    plt.xlim([0, 256])
    
    # Vykresleni legendy.
    plt.legend(('Histogram', 'Non-Maximum Suppression'), loc = 'upper left')
    
    # Vykresleni grafu.
    plt.show()   

    
if __name__ == '__main__':
    main()
    














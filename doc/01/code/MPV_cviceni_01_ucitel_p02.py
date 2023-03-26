# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 16:21:27 2014

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits: Ing. Marek Hruz, Ph.D.
@version: 2.0.0
"""


# Import OpenCV modulu.
import cv2 

# Import modulu pro pocitani s Pythonem.
import numpy as np 

# Import modulu pro vykreslovani grafu.
from matplotlib import pyplot as plt 


def non_maximum_suppression(hist, half_wnd):
    """
    Funkce pro nalezeni maxim.

    :param hist: Histogram nacteneho obrazku.
    :type hist: ndarray of int

    :param half_wnd: Velikost poloviny okenka, cele okenko = (2 * half_wnd) + 1.
    :type half_wnd: int

    :return out: Pole s oznacenim pozic nalezenych maxim. Nenulova hodnota na pozicich, kde bylo nalezeno maximum.
    :rtype out: ndarray of int
    """
    
    # Deklarace promennych.
    l = len(hist)
    out = np.zeros(len(hist), np.int8)
    max_idx = 0
    i = 0
    
    # Hlavni cyklus.
    while i < l:
        if max_idx < (i - half_wnd):
            max_idx = i - half_wnd

        e = min(i + half_wnd, (l - 1))
        
        while max_idx <= e:
            if hist[max_idx] > hist[i]:
                break

            max_idx += 1

        if max_idx > e:
            out[i] = 1
            max_idx = i + 1
            i += half_wnd

        i += 1
    
    return out


def multi_thresholding(img, max_positions):
    """
    Funkce pro prahovani vice prahy.

    :param img: Vstupni sedotonovy obrazek.
    :type img: ndarray of uint8

    :param max_positions: Pole s nenulovymi hodnotami na pozicich, kde byla nalezena maxima.
    :type max_positions: ndarray of int

    :return multi_threshold_img: Vystupni obrazek naprahovany N prahy.
    :rtype multi_threshold_img: ndarray of float
    """

    # Ziskani pozic maxim.
    idxs = []
    for i, x in enumerate(max_positions):
        if x != 0:
            idxs.append(i)

    thresholds = [0]

    # Vypocet hranic prahu.
    for i in range(len(idxs) - 1):
        thresholds.append(np.int((idxs[i] + idxs[i + 1]) / 2))

    thresholds.append(255)

    # Prahovani obrazku postupne vice prahy a nasledne spojeni do jednoho vysledneho obrazku.
    multi_threshold_img = np.zeros(np.shape(img))
    for i in range(len(thresholds) - 1):
        tmp_img = cv2.inRange(img, thresholds[i], thresholds[i + 1])
        tmp_img /= 255.0
        tmp_img *= idxs[i]
        multi_threshold_img += tmp_img

    return multi_threshold_img/255.0


def main():
    """
    Hlavni funkce volana po spusteni programu.    
    """
    
    # Nacteni sedotonoveho obrazku.
    img_gs = cv2.imread("./img/tsukuba_r.png", cv2.IMREAD_GRAYSCALE)
    
    # Vypocita histogram z obrazku.
    hist, bins = np.histogram(img_gs.flatten(), 256, [0, 256])
    
    # Volba velikosti poloviny okenka, cele okenko = (2 * half_wnd) + 1.
    half_wnd = 25
    
    # Volani funkce, ktera nalezne maxima v histogramu.
    # Vrati pole s nenulovymi hodnotami na pozicich, kde bylo
    # nalezeno maximum.
    max_positions = non_maximum_suppression(hist, half_wnd)

    # Vytvori novou figuru, stejne jako v MATLABu.
    plt.figure("Histogram")
    
    # Vykresleni sloupcu histogramu cervenou barvou.
    plt.hist(img_gs.flatten(), 256, [0, 256], color='r')

    # Prekresleni sloupcu histragramu, ve kterych bylo vyhodnoceno 
    # maximum, modrou barvou.
    for i in range(len(max_positions)):
        if max_positions[i] != 0:
            plt.bar(i, hist[i], width=1, color='b', hold=None)
    
    # Nastaveni limitu X-ove osy.
    plt.xlim([0, 256])
    
    # Vykresleni legendy.
    plt.legend(('Histogram', 'Non-Maximum Suppression'), loc='upper left')
    
    # Vykresleni grafu.
    plt.show()   

    # Volani funkce pro vypocet prahovani obrazku vice prahy.
    multi_threshold_img = multi_thresholding(img_gs, max_positions)

    # Vytvoreni pojmenovaneho okna.
    cv2.namedWindow("Multi-Thresholding")

    # Urceni, do jakeho okna se ma dany obrazek vykreslit.
    cv2.imshow("Multi-Thresholding", multi_threshold_img)

    # Nyni se obrazek vykresli na nekonecne dlouhou dobu v [ms], dokud
    # nebude stisknuta libovolna klavesa.
    cv2.waitKey(0)

if __name__ == '__main__':
    main()

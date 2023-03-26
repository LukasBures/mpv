# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 20:02:10 2014

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits: 
@version: 2.0.0
"""

import cv2  # Import OpenCV.
import numpy as np
import csv
import os


# -----------------------------------------------------------------------------
# Vypocte LBP kody, histogramy a FV.
# -----------------------------------------------------------------------------
def lbp(imggs):
    h, w = imggs.shape

    if h is not w or h is not 512 or w is not 512:
        print "Obrazek ma spatnou velikost:", w, "x", h
        return -1

    sz_cell = 128
    x_cell = w / sz_cell
    y_cell = h / sz_cell
    histograms = []

    for i in range(y_cell):
        for j in range(x_cell):
            nums = []
            for k in range(1, sz_cell - 1):
                for l in range(1, sz_cell - 1):
                    y_pos = (i * sz_cell) + k
                    x_pos = (j * sz_cell) + l

                    code = np.zeros((1, 8), np.int)

                    code[0, 0] = 128 if imggs[y_pos, x_pos] > imggs[y_pos - 1, x_pos - 1] else 0
                    code[0, 1] = 64 if imggs[y_pos, x_pos] > imggs[y_pos - 1, x_pos] else 0
                    code[0, 2] = 32 if imggs[y_pos, x_pos] > imggs[y_pos - 1, x_pos + 1] else 0
                    code[0, 3] = 16 if imggs[y_pos, x_pos] > imggs[y_pos, x_pos + 1] else 0
                    code[0, 4] = 8 if imggs[y_pos, x_pos] > imggs[y_pos + 1, x_pos + 1] else 0
                    code[0, 5] = 4 if imggs[y_pos, x_pos] > imggs[y_pos + 1, x_pos] else 0
                    code[0, 6] = 2 if imggs[y_pos, x_pos] > imggs[y_pos + 1, x_pos - 1] else 0
                    code[0, 7] = 1 if imggs[y_pos, x_pos] > imggs[y_pos, x_pos - 1] else 0

                    nums.append(np.sum(code))

            hist, _ = np.histogram(nums, 256, None, True, None, None)
            histograms = np.concatenate([histograms, hist])

    return histograms


# ------------------------------------------------------------------------------
# Priklad 1: Local binary patterns (LBP)
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    csv_path = './other/LBP.csv'

    if os.path.exists(csv_path):
        os.remove(csv_path)

    nImg = 40
    with open('./other/LBP.csv', 'wb') as csvfile:
        File = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for n in range(nImg):
            if n < 10:
                path = "./img/textures/0" + str(n) + ".png"
            else:
                path = "./img/textures/" + str(n) + ".png"

            print path

            # Nacteni barevneho obrazku.
            Img = cv2.imread(path, cv2.IMREAD_COLOR)

            # Prevede barevny obrazek do sedotonoveho formatu.
            ImgGS = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)

            FV = lbp(ImgGS)

            File.writerow(FV)

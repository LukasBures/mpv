# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 15:35:41 2014

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits: Ing. Marek Hruz, Ph.D., Ing. Petr Neduchal
@version: 2.0.0
"""

import cv2  # Import OpenCV.
import numpy as np
import csv
import time


# -----------------------------------------------------------------------------
# Vypocte LBP kody, histogramy a FV.
# -----------------------------------------------------------------------------
def lbp(imggs):
    h, w = imggs.shape

    if h != 512 or w != 512:
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


# -----------------------------------------------------------------------------
# Vrati indexy peti nejpodobnejsich textur.
# -----------------------------------------------------------------------------
def find_best_match(fv, lbp_code, n_best):
    [n_row, n_col] = lbp_code.shape
    lbp_code /= 16
    fv /= 16
    distance = np.zeros(n_row, np.float64)

    # ---------------------------------------------------------------------------
    # Euklidovska vzdalenost.
    # for i in range(n_row):
    #     for j in range(n_col):
    #         distance[i] = distance[i] + (np.power(fv[j], 2) - np.power(lbp_code[i, j], 2))
    #         distance[i] = np.sqrt(distance[i])

    # ---------------------------------------------------------------------------
    # Chi-square.
    for i in range(n_row):
        for j in range(n_col):
            if fv[j] - lbp_code[i, j] == 0 or fv[j] == 0:
                continue
            else:
                distance[i] = distance[i] + np.power((fv[j] - lbp_code[i, j]), 2) / fv[j]

    # ---------------------------------------------------------------------------
    # Histogram intersection.
    # for i in range(n_row):
    #    for j in range(n_col):
    #        if fv[j] < lbp_code[i, j]:
    #            distance[i] = distance[i] + fv[j]
    #        else:
    #            distance[i] = distance[i] + lbp_code[i, j]
    #
    # distance = 1 - distance

    # ---------------------------------------------------------------------------
    # for i in range(n_row):
    #    for j in range(n_col):
    #        if lbp_code[i, j] == 0 or fv[j] == 0:
    #            continue
    #        distance[i] = distance[i] + lbp_code[i, j] * np.log(lbp_code[i, j] / fv[j])

    # ---------------------------------------------------------------------------
    srt = np.sort(distance)
    best = np.zeros(n_best, np.int)
    for i in range(n_best):
        for j in range(len(distance)):
            if srt[i] == distance[j]:
                best[i] = j
    return best


# ------------------------------------------------------------------------------
# Priklad 1: Local binary patterns (LBP)
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    start = time.time()
    nImg = 40

    with open('./other/LBP.csv', 'rb') as csvfile:
        File = csv.reader(csvfile, delimiter=';', quotechar='|')
        r = 0
        LBPcode = np.zeros((nImg, 4096), np.float64)
        for row in File:
            LBPcode[r, :] = np.float64(row)
            r += 1

    # Databaze obrazku 0-39.
    nImg = np.random.randint(0, nImg)
    if nImg < 10:
        path = "./img/textures/0" + str(nImg) + ".png"
    else:
        path = "./img/textures/" + str(nImg) + ".png"

    # Nacte nahodny baervny obrazek a prevede ho do odstinu sedi.
    Img = cv2.imread(path, cv2.IMREAD_COLOR)
    ImgGS = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)

    # Zobrazi aktualne vybrany obrazek.
    cv2.namedWindow("Loaded image")
    cv2.imshow("Loaded image", Img)
    cv2.waitKey(1)

    # Spocita Feature Vector, zretezene histogramz LBP kodu.
    FV = lbp(ImgGS)

    if FV is not -1:
        # Vypocita nejblizsiho souseda pro 5 nejpodobnejsich textur.
        idx = find_best_match(FV, LBPcode, 5)

        # Vykresli 5 nejpodobnejsich textur.
        for p in range(len(idx)):
            if idx[p] < 10:
                path = "./img/textures/0" + str(idx[p]) + ".png"
            else:
                path = "./img/textures/" + str(idx[p]) + ".png"

            Img = cv2.imread(path, cv2.IMREAD_COLOR)

            cv2.namedWindow(str(p) + " - " + path)
            cv2.imshow(str(p) + " - " + path, Img)
            cv2.waitKey(1)
    else:
        print "LBP nebyl spravne vypocten!"

    end = time.time()
    print "Doba behu programu:", end - start, "[s]"
    cv2.waitKey()
    cv2.destroyAllWindows()

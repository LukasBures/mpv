# -*- coding: utf-8 -*-
"""
Created on 15:40 18.11.15

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits:
@version: 1.0.0
"""
import cv2
import numpy as np
import math
def normalise(PT):
    # vypocet stredni hodnoty a odchylky
    mu = np.array([np.mean(PT[0]), np.mean(PT[1])])
    a = np.std(PT,0)
    a = np.sqrt(2)/a
    # vzd = 0
    # for i in PT:
    #     vzd = vzd + np.linalg.norm(mu - i)
    # var = vzd/len(PT)

    # doplneni souradnic o homogenní souradnici
    PT2 = np.ones((len(PT), 3),dtype = np.float32 )
    for i in range(len(PT)):
        PT2[i, 0] = PT[i, 0]
        PT2[i, 1] = PT[i, 1]
    # transformacni(normlizacni matice)
    T = np.array([[a[0], 0, -1*a[0]*mu[0]],[0, a[1], -1*a[1]*mu[1]],[0, 0, 1]])
    # vypocet normalizovanych souradnic
    nPT = np.dot(PT2, T)
    return nPT, T


def find_fundamental_matrix(img_left, img_right):
    """
        Provede vypocet fundamentalni matice.
    :param img_left: Vstupni levy obrazek.
    :rtype img_left: 2D ndarray, uint8
    :param img_right: Vstupni pravy obrazek.
    :rtype img_right: 2D ndarray, uint8
    :return f: Vypoctena fundamentalni matice velikosti 3x3.
    :rtype f: 2D ndarray, float
    """
    # nalezení sift deskriptoru a keypointu v obou obrázcích
    sift = cv2.xfeatures2d.SIFT_create()
    KP_l, des_l = sift.detectAndCompute(img_left, None)
    KP_r, des_r = sift.detectAndCompute(img_right, None)

    # Matchování bodu v obrázcích
    index_params = dict(algorithm = 0, trees = 5)
    search_params = dict(checks = 50)
    matcher = cv2.BFMatcher()
    Matches = matcher.knnMatch(des_l, des_r, k = 2 )

    # Vybrání dobrých matchu
    GoodMatches = []
    for m, n in Matches:
        if m.distance < 0.7 * n.distance:
            GoodMatches.append(m)

    # normalizace bodu
    PT_l = np.float32([KP_l[m.queryIdx].pt for m in GoodMatches]).reshape(-1, 2)
    PT_r = np.float32([KP_r[m.trainIdx].pt for m in GoodMatches]).reshape(-1, 2)
    Pl, T1 = normalise(PT_l)
    Pr, T2 = normalise(PT_r)

    # vytvo?ení matice A pro výpo?et matice F
    A = list()

    for ind, x in enumerate(Pl):
        y = Pr[ind]
        A.append([x[0]*y[0], x[0]*y[1], x[0], x[1]*y[0], x[1]*y[1], x[1], x[0], x[1], 1])
    A = np.array(A)

    # výpo?et F pomoci SVD rozkladu
    U, s, V = np.linalg.svd(A)
    V = V.conj().T;
    F = V[:,8].reshape(3,3).copy()
    while np.linalg.matrix_rank(F)!= 2:
        (U,D,V) = np.linalg.svd(F);
        F = np.dot(np.dot(U,np.diag([D[0], D[1], 0])),V);
    F = np.dot(np.dot(T2.T,F),T1.T);
    print F
    return F



# img = cv2.imread('l1.jpg', cv2.COLOR_BGR2GRAY)
# img2 = cv2.imread('r1.jpg', cv2.COLOR_BGR2GRAY)
# find_fundamental_matrix(img, img2)

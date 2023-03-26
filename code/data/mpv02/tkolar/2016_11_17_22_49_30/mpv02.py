# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 13:38:32 2014

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits:
@version: 2.0.0
"""
import cv2
import glob, os, random
import numpy as np
# ------------------------------------------------------------------------------
def klas(tridy, obraz):
    podobnost = list()
    for i in tridy:
        podobnost.append( np.arccos( np.dot(i,obraz) / ( np.linalg.norm(i) * np.linalg.norm(obraz) ) ) ) #vypocet uhlu, který svírají vektory histogramu
    trida = np.argmin(podobnost)
    return trida

def NN(deskr, stredy):
    hist = np.zeros(len(stredy))
    for i in deskr:
        vzd = list()
        for j in stredy:
            vzd = np.append(vzd, np.linalg.norm(i-j))
        a = np.argmin(vzd)
        hist[a] += 1

    return hist

def k_means(trid, deskr):

    stredy = list()
    for i in range(trid):
        stredy.append(random.choice(deskr))

    stredy = list()
    stredy.append(random.choice(deskr))

    for x in range(trid-1):
        vzd = np.zeros(len(deskr))
        for i , j in enumerate(deskr):
            for k in stredy:
                vzd[i] += np.linalg.norm(deskr[i]-k)
        m = np.argmax(vzd)
        stredy.append(deskr[m])


    pocet = list()
    old_stredy = np.zeros(trid)
    trida = list()
    while True:
        if np.array_equal(old_stredy, stredy):
            break
        vzd = list()

        for i in range(trid):   #pocitani vzdalenosti ke stredu tridy
            a = list()
            for j in range(len(deskr)):
                a.append(np.linalg.norm(deskr[j]-stredy[i])) #euklidovska vzdalenost
            vzd.append(a)
        vzd = np.transpose(np.asarray(vzd))
        for i in range(len(deskr)): #urceni tridy
            trida.append(np.argmin(vzd[i]))
        deskr = np.asarray(deskr)
        suma = np.zeros([trid, deskr.shape[1]])
        pocet = np.zeros(trid)

        for i in range(len(deskr)):
            suma[trida[i]] = np.add(suma[trida[i]],deskr[i])
            pocet[trida[i]] += 1
        old_stredy = stredy
        for i in range(trid):
            stredy[i] = suma[i]/pocet[i] #vypocet novych stredu

    return stredy, trida


def bow(train_data, train_label, test_data, n_visual_words):
    """
    Provede klasifikaci pomoci BoW algoritmu.

    :param train_data: Vstupni list trenovacich sedotonovych obrazku.
    :rtype train_data: list of 2D ndarray, uint8

    :param train_label: Vstupni list trid, do kterych patri trenovaci sedotonove obrazky.
    :rtype train_label: list of int

    :param test_data: Vstupni list testovacich sedotonovych obrazku.
    :rtype test_data: list of 2D ndarray, uint8

    :param n_visual_words: Pocet vizualnich slov.
    :rtype n_visual_words: int

    :return cls: Vystupni list trid odpovidajicich obrazkum v test_data.
    :rtype cls: list of int
    """
    print n_visual_words
    lab = list()
    for i in train_label:
        a = 125 * [i]
        lab.extend(a)
    cls = list()
    #trenování
    print ('SIFTY')
    sift = cv2.xfeatures2d.SIFT_create(125)
    deskr = list()
    for i in train_data: #vypocet SIFT deskriptoru pro trenovani
        a, des = sift.detectAndCompute(i, None)
        deskr.extend(des[:125])
    print('Start K-means')
    stredy, trida = k_means(n_visual_words, deskr) # clusterovani pomoci k-means
    print('Konec k-means')
    hist = np.zeros([n_visual_words, n_visual_words])
    for i in range(len(deskr)): #vypocet modelovych histogramu trid
        hist[trida[i],lab[i]] += 1
    for i in range(n_visual_words):
        a = sum(hist[i])
        for j in range(n_visual_words):
            hist[i, j] = hist[i, j]/a
    #klasifikace
    for i in test_data:
        _, des = sift.detectAndCompute(i, None) #vypocet SIFT deskriptoru pro klasifikaci
        hist_n = NN(des[:125], stredy) # vypocet histogramu klasifikovaneho obrazu
        suma = sum(hist_n)
        for j in range(n_visual_words):
            hist_n[j] = hist_n[j]/ suma
        cls.append(klas(hist, hist_n))

    print cls
    return cls






#zakomentovat!!!!!!!!!!!!!!! nebo smazat
# tr = list()
# label = list()
# test = list()
# n = 5
# trida = 0
# for root, dirs, files in os.walk("./train"):
#     for file in files:
#         if file.endswith(".jpg"):
#              tr.append(cv2.imread(os.path.join(root, file), cv2.IMREAD_GRAYSCALE))
#              label.append(trida)
#         if file.endswith("40.jpg"):
#             trida += 1
#             test.extend(tr[-5:])
#             del tr[-5:]
#             del label[-5:] #bordel na testovaní
#             break
#
#
# bow(tr, label, test, n)

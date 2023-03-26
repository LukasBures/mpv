# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 10:32:00 2014

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits: Ing. Marek Hruz, Ph.D.
@version: 3.0.0

Revision Note:
3.0.0 - 26.9.2016 - Updated for OpenCV 3.1.0 version
"""

import cv2  # Import OpenCV modulu.
import numpy as np  # Import modulu pro pocitani s Pythonem.
import time  # Import modulu pro mereni casu.


# ----------------------------------------------------------------------------------------------------------------------
# Priklad 1: Shlukovaci algoritmus Mean-Shift.

# Nacteni barevneho obrazku.
Img = cv2.imread("./img/mms.jpg", cv2.IMREAD_COLOR)

# Vytvoreni pojmenovanenych oken.
cv2.namedWindow("Original Image", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Hue", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Saturation", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("Value", cv2.WINDOW_AUTOSIZE)
cv2.moveWindow("Original Image", 100, 100)
cv2.moveWindow("Hue", 100, 100)
cv2.moveWindow("Saturation", 100, 100)
cv2.moveWindow("Value", 100, 100)

# Zjisti velikost obrazku.
height, width, depth = Img.shape

# Prevede obrazek z barevneho prostoru BGR do HSV.
ImgHSV = cv2.cvtColor(Img, cv2.COLOR_BGR2HSV_FULL)

# Urceni, do jakeho okna se ma dany obrazek vykreslit.
cv2.imshow("Original Image", Img)
cv2.imshow("Hue", ImgHSV[:, :, 0])
cv2.imshow("Saturation", ImgHSV[:, :, 1])
cv2.imshow("Value", ImgHSV[:, :, 2])

# Nyni se obrazky vykresli, prodleni 1 [ms].
cv2.waitKey(1)

# ----------------------------------------------------------------------------------------------------------------------
# Priprava dat, preprocessing.

# Ulozeni aktualniho casu pro nasledne mereni doby behu jednotlivych casti programu.
totalStart = time.time()
start = time.time()

# Definice promennych. 
DataList = []
weight = []
DataImg = np.zeros((256, 256), np.uint8)

# Vypocte histogram, aby se zjistila informace o poctu jednotlivych kombinaci HS hodnot v obrazku.
Hist, _xedges, _yedges = np.histogram2d(ImgHSV[:, :, 1].ravel(), ImgHSV[:, :, 0].ravel(),[256, 256],
                                        [[0, 256], [0, 256]])

# Vyber dat z histogramu do listu. Prevedeni z 2D struktury na 1D.
for i in range(256):
    for j in range(256):
        if Hist[i][j] > 0:
            DataList.append([j, i])
            weight.append(Hist[i][j])

# Prevede listy na numpy array.
DataArray = np.array(DataList, np.float)
weight = np.array(weight, np.float)

# Zmeri cas predzpracovani a vypise ho do konzole.
end = time.time()
print "Time of preprocessing =", (end - start), "[s]"
print

# ----------------------------------------------------------------------------------------------------------------------
# Mean-Shift algoritmus.

# Definice promennych.
sWin = 50  # Volba velikosti prumeru kruhoveho okenka.
sWinHalf = sWin / 2  # Polomer.
nCluster = 1  # Cislo shluku.
bagOfCenters = []  # Ulozeni [cislo, [xStred, yStred]] shluku.
nthIterace = 0  # Inkrementacni promenna pro pocitani iteraci.
clusterVote = {}  # Slovnik pro uchovavani informace o hlasech jednotlivych bodu.

# Naplneni slovniku prazdnym listem.
for i in range(len(weight)):
    clusterVote[i] = []

# Cyklus, ktery je zastaven pokud neexistuje bod, ktery není zarazeny do nejake mnoziny.
while True:

    # Inkrementace iterace a jeji vypsani do konzole.
    nthIterace += 1
    if nthIterace == 1:
        print "Number of iteration =", nthIterace, ",",
    else:
        print nthIterace, ",",

    # Startovaci bod (stred okenka), Actual Center, nastavi.
    aCenter = []
    for e in range(len(DataArray)):
        if not clusterVote[e]:
            aCenter = DataArray[e]
            break

    # Pokud neexistuje zadny nezarazeny bod ukonci algoritmus.
    if len(aCenter) == 0:
        break

    # Aktualni bod stredu.
    aPts = []
    boolDistanceAll = np.zeros(len(DataArray), np.bool)

    # Cyklus dokud nedokonverguje stred.
    while True:

        # Vzdalenost vsech bodu od aktualniho stredu.
        distance = np.sqrt(np.sum(np.power(np.subtract(aCenter, DataArray), 2), 1))

        # Ulozi True na pozici, pokud je bod v okenku kolem aktualniho stredu.
        boolDistance = distance <= sWinHalf

        # Logicky OR, uchovava informaci o vsech navstivenych bodech behem behem
        # jedne cesty.
        boolDistanceAll = boolDistanceAll | boolDistance

        # Vypocte vazeny soucet bodu, ktere jsou v aktualnim okenku.
        sumX = 0.0
        sumY = 0.0
        nPt = 0.0
        for i in range(len(boolDistance)):
            if boolDistance[i]:
                sumX = sumX + (DataArray[i, 0] * weight[i])
                sumY = sumY + (DataArray[i, 1] * weight[i])
                nPt = nPt + weight[i]

        # Nastavi stred na stary stred.
        oldCenter = aCenter

        # Vypocte novy stred, kam se algoritmus posune.
        aCenter = np.array((sumX / nPt, sumY / nPt))

        # Zastavovaci podminka, pokud jsou polohy stredu ve dvou po sobe jdoucich
        # iteracich stejne.
        if (oldCenter == aCenter).all():

            # Pokud nemam ulozeny zadny stred.
            if len(bagOfCenters) == 0:

                # Uloz [cislo clusteru, [stred clusteru X, Y]]
                bagOfCenters.append([nCluster, aCenter])

                # Uloz hlas shluku pro body, ktere byly cestou navstiveny.
                for i in range(len(boolDistanceAll)):
                    if boolDistanceAll[i]:
                        clusterVote[i].append(nCluster)

            # Pokud jiz existuji nejake ulozene stredy shluku.
            else:
                # Promenna urcijici, jestli dany stred jiz existuje.
                same = False

                # Projde vsechny existujici stredy.
                for c in range(len(bagOfCenters)):
                    if (bagOfCenters[c][1] == aCenter).all():
                        # Pokud existuje stred.
                        same = True

                        # Uloz hlas shluku pro body, ktere byly cestou
                        # navstiveny.
                        for i in range(len(boolDistanceAll)):
                            if boolDistanceAll[i]:
                                clusterVote[i].append(bagOfCenters[c][0])

                # Pokud dany stred neexistuje.
                if not same:
                    # Vytvor novy shluk (nove cislo shluku).
                    nCluster += 1

                    # Uloz [cislo clusteru, [stred clusteru X, Y]]
                    bagOfCenters.append([nCluster, aCenter])

                    # Uloz hlas shluku pro body, ktere byly cestou navstiveny.
                    for i in range(len(boolDistanceAll)):
                        if boolDistanceAll[i]:
                            clusterVote[i].append(nCluster)
            break

# ----------------------------------------------------------------------------------------------------------------------
# Postprocessing.

# Vypisy do konzole.
print "end of iterations!"
print
print "Postprocessing and drawing ...",

# Vybere nejpocetnejsi mnozinu hlasu a priradi bod do clusteru
clusterAssign = np.zeros(len(DataArray), np.uint8)
for i in range(len(DataArray)):
    clusterAssign[i] = max(set(clusterVote[i]), key=clusterVote[i].count)

# Zaokrouhleni a pretypovani stredu.
for i in range(len(bagOfCenters)):
    bagOfCenters[i][1] = np.uint8(np.round(bagOfCenters[i][1]))

PtClD = {}  # Slovnik {bod x, y: prirazeny shluk}
DataArray = np.uint8(DataArray)  # Pretypovani.
# Naplni slovnik.
for i in range(len(DataArray)):
    PtClD[DataArray[i][0], DataArray[i][1]] = clusterAssign[i]

# Tvorba vysledneho obrazku shluku.
clusterResult = np.zeros((256, 256, 3), np.uint8)
V = 200  # Nastaveni hodnoty Value na 200.
for i in range(len(clusterAssign)):
    Hval = bagOfCenters[PtClD[DataArray[i][0], DataArray[i][1]] - 1][1][0]
    Sval = bagOfCenters[PtClD[DataArray[i][0], DataArray[i][1]] - 1][1][1]
    clusterResult[DataArray[i][1], DataArray[i][0], :] = [Hval, Sval, V]

# Prevedeni nazpet z HSV do BGR prostoru a nasledne vykresleni.
clusterResult = cv2.cvtColor(clusterResult, cv2.COLOR_HSV2BGR)
cv2.namedWindow("Clustering Result", cv2.WINDOW_AUTOSIZE)
cv2.moveWindow("Clustering Result", 100, 100)
cv2.imshow("Clustering Result", clusterResult)
cv2.waitKey(1)

# ----------------------------------------------------------------------------------------------------------------------
# Vykresleni.

start = time.time()
ImgResult = np.zeros((height, width, depth), np.uint8)
ImgResultOriginalV = np.zeros((height, width, depth), np.uint8)
for i in range(height):
    for j in range(width):
        x = ImgHSV[i, j, 0]  # Hue
        y = ImgHSV[i, j, 1]  # Saturation
        z = ImgHSV[i, j, 2]  # Value
        # HSV
        ImgResult[i, j, :] = [bagOfCenters[PtClD[x, y] - 1][1][0], bagOfCenters[PtClD[x, y] - 1][1][1], V]
        ImgResultOriginalV[i, j, :] = [bagOfCenters[PtClD[x, y] - 1][1][0], bagOfCenters[PtClD[x, y] - 1][1][1], z]

# Vykresleni originalniho obrazku po shlukovani, kazdy shluk je reprezentovan barvou stredu shluku. Pro hodnotu Value
# byla zvolena hodnota V viz vyse.
ImgResult = cv2.cvtColor(ImgResult, cv2.COLOR_HSV2BGR)
cv2.namedWindow("Result", cv2.WINDOW_AUTOSIZE)
cv2.moveWindow("Result", 100, 100)
cv2.imshow("Result", ImgResult)

# Vykresleni originalniho obrazku po shlukovani, kazdy shluk je reprezentovan barvou stredu shluku. Pro hodnotu Value
# byla zvolena hodnota originalniho obrazku.
ImgResultOriginalV = cv2.cvtColor(ImgResultOriginalV, cv2.COLOR_HSV2BGR)
cv2.namedWindow("Result with original Value", cv2.WINDOW_AUTOSIZE)
cv2.moveWindow("Result with original Value", 100, 100)
cv2.imshow("Result with original Value", ImgResultOriginalV)


# Vypisy do konzole.
end = time.time()
print "time of drawing =", (end - start), "[s]"
print
print "Total time =", (end - totalStart), "[s]"

# Nyni se obrazek vykresli na nekonecne dlouhou dobu v [ms], dokud nebude stisknuta libovolna klavesa.
cv2.waitKey(0)

# Zavre vsechna vytvorena OpenCV okna.
cv2.destroyAllWindows()

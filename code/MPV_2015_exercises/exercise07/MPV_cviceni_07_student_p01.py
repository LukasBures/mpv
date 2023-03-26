# -*- coding: utf-8 -*-
"""
Created on Tue Nov 04 17:31:06 2014

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@version: 2.0.0
"""

import cv2  # Import OpenCV modulu.
import numpy as np  # Import modulu pro pocitani s Pythonem.
import time  # Import modulu pro mereni casu.


def print_help():
    print "Help:"
    print "Face detect: yellow rectangle."
    print "Camshift tracking: purple rectangle."
    print "You can start to train camshift model with 'r' key."


def mean_shift(img, result_img):
    print "-------------------------------------------------------------------"
    # Zjisti velikost obrazku.
    height, width, depth = img.shape

    # Prevede obrazek z barevneho prostoru BGR do HSV.
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)

    # Nyni se obrazky vykresli, prodleni 1 [ms].
    cv2.waitKey(1)

    # ------------------------------------------------------------------------------
    # Priprava dat, preprocessing.

    # Ulozeni aktualniho casu pro nasledne mereni doby behu jednotlivych casti
    # programu.
    total_start = time.time()
    start = time.time()

    # Definice promennych.
    datalist = []
    weight = []
    # DataImg = np.zeros((256, 256), np.uint8)

    # Vypocte histogram, aby se zjistila informace o poctu jednotlivych kombinaci
    # HS hodnot v obrazku.
    hist, _, _ = np.histogram2d(imghsv[:, :, 1].ravel(), imghsv[:, :, 0].ravel(),
                                [256, 256], [[0, 256], [0, 256]])

    # Vyber dat z histogramu do listu. Prevedeni z 2D struktury na 1D.
    for i in range(256):
        for j in range(256):
            if hist[i][j] > 0:
                datalist.append([j, i])
                weight.append(hist[i][j])

    # Prevede listy na numpy array.
    dataarray = np.array(datalist, np.float)
    weight = np.array(weight, np.float)

    # Zmeri cas predzpracovani a vypise ho do konzole.
    end = time.time()
    print "Time of preprocessing =", (end - start), "[s]"
    print

    # ------------------------------------------------------------------------------
    # Mean-Shift algoritmus.

    # Definice promennych.
    s_win = 40  # Volba velikosti prumeru kruhoveho okenka.
    s_win_half = s_win / 2  #
    n_cluster = 1  # Cislo shluku.
    bag_of_centers = []  # Ulozeni [cislo, [xStred, yStred]] shluku.
    nth_iter = 0  # Inkrementacni promenna pro pocitani iteraci.
    cluster_vote = {}  # Slovnik pro uchovavani informace o hlasech jednotlivych bodu.

    # Naplneni slovniku prazdnym listem.
    for i in range(len(weight)):
        cluster_vote[i] = []

    # Cyklus, ktery je zastavem pokud neexistuje bod, ktery není zarazeny do nejake
    # mnoziny.
    while True:

        # TODO: Implementace Mean-shift
        break

    # ------------------------------------------------------------------------------
    # Postprocessing.

    # Vypisy do konzole.
    print "end of iterations!"
    print
    print "Postprocessing and drawing ...",

    # Vybere nejpocetnejsi mnozinu hlasu a priradi bod do clusteru
    cluster_assign = np.zeros(len(dataarray), np.uint8)
    for i in range(len(dataarray)):
        cluster_assign[i] = max(set(cluster_vote[i]), key=cluster_vote[i].count)

    sz_of_cluster = np.zeros((n_cluster + 1, 1), np.int)  # NEBER v potaz 0 cluster !!!
    for i in range(len(cluster_assign)):
        sz_of_cluster[cluster_assign[i]] += 1

    max_sz = np.max(sz_of_cluster)
    num_of_max_cluster = 0
    for i in range(1, len(sz_of_cluster)):
        if max_sz == sz_of_cluster[i]:
            num_of_max_cluster = i
            break

    for i in range(len(dataarray)):
        if cluster_assign[i] == num_of_max_cluster:
            result_img[dataarray[i][1], dataarray[i][0] - 1] += 1.0

    # Zaokrouhleni a pretypovani stredu.
    for i in range(len(bag_of_centers)):
        bag_of_centers[i][1] = np.uint8(np.round(bag_of_centers[i][1]))

    pt_cld = {}  # Slovnik {bod x, y: prirazeny shluk}
    dataarray = np.uint8(dataarray)  # Pretypovani.
    # Naplni slovnik.
    for i in range(len(dataarray)):
        pt_cld[dataarray[i][0], dataarray[i][1]] = cluster_assign[i]

    # Tvorba vysledneho obrazku shluku.
    cluster_result = np.zeros((256, 256, 3), np.uint8)
    v_value = 200  # Nastaveni hodnoty Value na 200.
    for i in range(len(cluster_assign)):
        h_val = bag_of_centers[pt_cld[dataarray[i][0], dataarray[i][1]] - 1][1][0]
        s_val = bag_of_centers[pt_cld[dataarray[i][0], dataarray[i][1]] - 1][1][1]
        cluster_result[dataarray[i][1], dataarray[i][0], :] = [h_val, s_val, v_value]

    # Prevedeni nazpet z HSV do BGR prostoru a nasledne vykresleni.
    cluster_result = cv2.cvtColor(cluster_result, cv2.COLOR_HSV2BGR)
    cv2.imshow("Clustering Result", cluster_result)
    cv2.waitKey(1)

    # ------------------------------------------------------------------------------
    # Vykresleni.

    start = time.time()
    img_result = np.zeros((height, width, depth), np.uint8)
    img_result_original_v = np.zeros((height, width, depth), np.uint8)
    for i in range(height):
        for j in range(width):
            xx = imghsv[i, j, 0]  # Hue
            yy = imghsv[i, j, 1]  # Saturation
            zz = imghsv[i, j, 2]  # Value
            # HSV
            img_result[i, j, :] = [bag_of_centers[pt_cld[xx, yy] - 1][1][0],
                                   bag_of_centers[pt_cld[xx, yy] - 1][1][1], v_value]

            img_result_original_v[i, j, :] = [bag_of_centers[pt_cld[xx, yy] - 1][1][0],
                                              bag_of_centers[pt_cld[xx, yy] - 1][1][1], zz]

    # Vykresleni originalniho obrazku po shlukovani, kazdy shluk je reprezentovan
    # barvou stredu shluku. Pro hodnotu Value byla zvolena hodnota V viz vyse.
    img_result = cv2.cvtColor(img_result, cv2.COLOR_HSV2BGR)
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Result", img_result)

    # Vykresleni originalniho obrazku po shlukovani, kazdy shluk je reprezentovan
    # barvou stredu shluku. Pro hodnotu Value byla zvolena hodnota originalniho
    # obrazku.
    img_result_original_v = cv2.cvtColor(img_result_original_v, cv2.COLOR_HSV2BGR)
    cv2.namedWindow("Result with original Value", cv2.WINDOW_NORMAL)
    cv2.imshow("Result with original Value", img_result_original_v)

    # Vypisy do konzole.
    end = time.time()
    print "time of drawing =", (end - start), "[s]"
    print
    print "Total time =", (end - total_start), "[s]"
    print "-------------------------------------------------------------------"

    # Nyni se obrazek vykresli na nekonecne dlouhou dobu v [ms], dokud nebude
    # stisknuta libovolna klavesa.
    cv2.waitKey(1)

    return result_img


# ------------------------------------------------------------------------------
# Priklad 1:
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    print_help()

    faceDetector = cv2.CascadeClassifier("./other/haarcascade_frontalface_alt2.xml")

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Original Video", 1)
    cv2.namedWindow("Clustering Result", 0)
    cv2.namedWindow("ResultImg", 0)

    ResultImg = np.zeros((256, 256), np.float)
    nTrainImgs = 5
    nImgs = 0
    normalize = False
    detect = True
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, Img = cap.read()
    window = (0, 0, 0, 0)
    if ret:
        window = (0, 0, Img.shape[1], Img.shape[0])

    while True:
        ret, Img = cap.read()

        if ret:
            if detect:
                ImgGS = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
                faces = faceDetector.detectMultiScale(ImgGS, 1.3, 5)
                ImgRec = np.copy(Img)

                for (x, y, w, h) in faces:
                    roiColor = Img[y:y + h, x:x + w]
                    cv2.rectangle(ImgRec, (x, y), (x + w, y + h), (0, 255, 255), 2)

                cv2.imshow("Original Video", ImgRec)

            if normalize:
                ImgGS = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
                faces = faceDetector.detectMultiScale(ImgGS, 1.3, 5)
                ImgRec = np.copy(Img)

                for (x, y, w, h) in faces:
                    roiColor = Img[y:y + h, x:x + w]
                    cv2.rectangle(ImgRec, (x, y), (x + w, y + h), (0, 255, 255), 2)

                cv2.imshow("Original Video", ImgRec)

                if len(faces) > 0:
                    cv2.imshow("Face", roiColor)

                    if nImgs < nTrainImgs:
                        ResultImg = mean_shift(roiColor, ResultImg)
                        print "Trenovaci obrazek: ", nImgs + 1, "z", nTrainImgs
                        nImgs += 1

                    elif nImgs == nTrainImgs and normalize:
                        print "Normalizuji"
                        print "-------------------------------------------------------------------"
                        print_help()
                        ma = np.max(ResultImg)
                        ResultImg = (ResultImg / ma)
                        normalize = False
                        ResultImg = np.float32(ResultImg)
                        cv2.imshow("ResultImg", ResultImg)

            elif not normalize and not detect:
                cv2.destroyWindow("Clustering Result")
                cv2.destroyWindow("Result with original Value")
                cv2.destroyWindow("Result")
                cv2.destroyWindow("Face")

                ImgHSV = cv2.cvtColor(Img, cv2.COLOR_BGR2HSV_FULL)
                bImg = np.zeros(Img.shape, np.float32)
                bImg = ResultImg[ImgHSV[:, :, 1], ImgHSV[:, :, 0]]
                cv2.imshow("Back Projection", bImg)
                cv2.imshow("Original Video", Img)

                if window == (0, 0, 0, 0):
                    window = (0, 0, Img.shape[1], Img.shape[0])

                ImgRec = np.copy(Img)

                # --------------------------------------------------------------
                # Camshift
                ret, window = cv2.CamShift(bImg, window, criteria)
                if ret:
                    cv2.rectangle(ImgRec, (window[0], window[1]), (window[0] + window[2], window[1] + window[3]),
                                  (255, 0, 255), 2)

                # --------------------------------------------------------------
                # Detekce obliceje
                ImgGS = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
                faces = faceDetector.detectMultiScale(ImgGS, 1.3, 5)

                for (x, y, w, h) in faces:
                    roiColor = Img[y:y + h, x:x + w]
                    cv2.rectangle(ImgRec, (x, y), (x + w, y + h), (0, 255, 255), 2)
                # --------------------------------------------------------------

                cv2.imshow("Original Video", ImgRec)

        key = cv2.waitKey(1)

        if key == 27:  # esc
            break
        elif key & 0xFF == ord('r'):  # 'r' key
            ResultImg = np.zeros((256, 256), np.int)
            nImgs = 0
            normalize = True
            detect = False

    cap.release()
    cv2.destroyAllWindows()

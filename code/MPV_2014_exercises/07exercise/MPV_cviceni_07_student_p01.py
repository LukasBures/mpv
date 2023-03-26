# -*- coding: utf-8 -*-
"""
Created on Fri Nov 07 09:48:45 2014

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@version: 1.0.0
"""


import cv2 # Import OpenCV modulu.
import numpy as np # Import modulu pro pocitani s Pythonem.
import time # Import modulu pro mereni casu.

def printHelp():
    print "Help:"
    print "Face detect: yellow rectangle."
    print "Camshift tracking: purple rectangle."
    print "You can start to train camshift model with 'r' key."

def MeanShift(Img, ResultImg):
    print "-------------------------------------------------------------------"
    # Zjisti velikost obrazku.
    height, width, depth = Img.shape
    
    # Prevede obrazek z barevneho prostoru BGR do HSV.
    ImgHSV = cv2.cvtColor(Img, cv2.COLOR_BGR2HSV_FULL) 
    
    # Nyni se obrazky vykresli, prodleni 1 [ms].
    cv2.waitKey(1)
    
    #------------------------------------------------------------------------------
    # Priprava dat, preprocessing.
    
    # Ulozeni aktualniho casu pro nasledne mereni doby behu jednotlivych casti
    # programu.
    totalStart = time.time()
    start = time.time()
    
    # Definice promennych. 
    DataList = []
    weight = []
    #DataImg = np.zeros((256, 256), np.uint8)
    
    # Vypocte histogram, aby se zjistila informace o poctu jednotlivych kombinaci
    # HS hodnot v obrazku.
    Hist, _, _ = np.histogram2d(ImgHSV[:, :, 1].ravel(), ImgHSV[:, :, 0].ravel(),
                                [256, 256], [[0, 256], [0, 256]])
                                
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
    
    #------------------------------------------------------------------------------
    # Mean-Shift algoritmus.
    
    # Definice promennych.
    sWin = 40 # Volba velikosti prumeru kruhoveho okenka.
    sWinHalf = sWin / 2 # 
    nCluster = 1 # Cislo shluku.
    bagOfCenters = [] # Ulozeni [cislo, [xStred, yStred]] shluku.
    nthIterace = 0 # Inkrementacni promenna pro pocitani iteraci.
    clusterVote = {} #Slovnik pro uchovavani informace o hlasech jednotlivych bodu.
    
    # Naplneni slovniku prazdnym listem.
    for i in range(len(weight)):
        clusterVote[i] = []
    
    # Cyklus, ktery je zastavem pokud neexistuje bod, ktery není zarazeny do nejake
    # mnoziny.
    while(True):
        














    # Vase implementace








        break












    # Vase implementace

























    # Vase implementace





















    # Vase implementace




















    
    #------------------------------------------------------------------------------
    # Postprocessing.
    
    # Vypisy do konzole.
    print "end of iterations!"
    print 
    print "Postprocessing and drawing ...",
    
    # Vybere nejpocetnejsi mnozinu hlasu a priradi bod do clusteru
    clusterAssign = np.zeros(len(DataArray), np.uint8)
    for i in range(len(DataArray)):
        clusterAssign[i] = max(set(clusterVote[i]), key=clusterVote[i].count)
        
    szOfCluster = np.zeros((nCluster + 1, 1), np.int) # NEBER v potaz 0 cluster !!!
    for i in range(len(clusterAssign)):
        szOfCluster[clusterAssign[i]] += 1
    
    maxSz = np.max(szOfCluster)
    for i in range(1, len(szOfCluster)):
        if(maxSz == szOfCluster[i]):
            numOfMaxCluster = i
            break

    for i in range(len(DataArray)):
        if(clusterAssign[i] == numOfMaxCluster):
            ResultImg[DataArray[i][1], DataArray[i][0] - 1] += 1.0
        











    
    # Zaokrouhleni a pretypovani stredu.
    for i in range(len(bagOfCenters)):
        bagOfCenters[i][1] = np.uint8(np.round(bagOfCenters[i][1]))

    PtClD = {} # Slovnik {bod x, y: prirazeny shluk}
    DataArray = np.uint8(DataArray) # Pretypovani.
    # Naplni slovnik.
    for i in range(len(DataArray)):
        PtClD[DataArray[i][0], DataArray[i][1]] = clusterAssign[i]
    
    # Tvorba vysledneho obrazku shluku.
    clusterResult = np.zeros((256, 256, 3), np.uint8)
    V = 200 # Nastaveni hodnoty Value na 200.
    for i in range(len(clusterAssign)):
        Hval = bagOfCenters[PtClD[DataArray[i][0], DataArray[i][1]] - 1][1][0]
        Sval = bagOfCenters[PtClD[DataArray[i][0], DataArray[i][1]] - 1][1][1]
        clusterResult[DataArray[i][1], DataArray[i][0], :] = [Hval, Sval, V]
    
    # Prevedeni nazpet z HSV do BGR prostoru a nasledne vykresleni.
    clusterResult = cv2.cvtColor(clusterResult, cv2.COLOR_HSV2BGR)
    cv2.imshow("Clustering Result", clusterResult)
    cv2.waitKey(1)
    
    #------------------------------------------------------------------------------
    # Vykresleni.
    
    start = time.time()
    ImgResult = np.zeros((height, width, depth), np.uint8)
    ImgResultOriginalV = np.zeros((height, width, depth), np.uint8)
    for i in range(height):
        for j in range(width):
            x = ImgHSV[i, j, 0] # Hue
            y = ImgHSV[i, j, 1] # Saturation
            z = ImgHSV[i, j, 2] # Value
            # HSV
            ImgResult[i, j, :] = [bagOfCenters[PtClD[x, y] - 1][1][0],
                                  bagOfCenters[PtClD[x, y] - 1][1][1], V]
                                  
            ImgResultOriginalV[i, j, :] = [bagOfCenters[PtClD[x, y] - 1][1][0],
                                           bagOfCenters[PtClD[x, y] - 1][1][1], z]
    
    # Vykresleni originalniho obrazku po shlukovani, kazdy shluk je reprezentovan
    # barvou stredu shluku. Pro hodnotu Value byla zvolena hodnota V viz vyse.
    ImgResult = cv2.cvtColor(ImgResult, cv2.COLOR_HSV2BGR)
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Result", ImgResult)
    
    # Vykresleni originalniho obrazku po shlukovani, kazdy shluk je reprezentovan
    # barvou stredu shluku. Pro hodnotu Value byla zvolena hodnota originalniho
    # obrazku.
    ImgResultOriginalV = cv2.cvtColor(ImgResultOriginalV, cv2.COLOR_HSV2BGR)
    cv2.namedWindow("Result with original Value", cv2.WINDOW_NORMAL)
    cv2.imshow("Result with original Value", ImgResultOriginalV)
    
    
    # Vypisy do konzole.
    end = time.time()
    print "time of drawing =", (end - start), "[s]"
    print 
    print "Total time =", (end - totalStart), "[s]"
    print "-------------------------------------------------------------------"
    
    # Nyni se obrazek vykresli na nekonecne dlouhou dobu v [ms], dokud nebude
    # stisknuta libovolna klavesa.
    cv2.waitKey(1)

    return ResultImg


#------------------------------------------------------------------------------
# Priklad 1: 
#------------------------------------------------------------------------------
if __name__ == '__main__':
    printHelp()
    
    faceDetector = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")    
    
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
    if(ret):
        window = (0, 0, Img.shape[1], Img.shape[0])
        
    while(True):
        ret, Img = cap.read()       
        
        if(ret):
            if(detect == True):
                ImgGS = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
                faces = faceDetector.detectMultiScale(ImgGS, 1.3, 5)
                ImgRec = np.copy(Img)
                
                for (x, y, w, h) in faces:
                    roiColor = Img[y:y + h, x:x + w]                
                    cv2.rectangle(ImgRec, (x, y) ,(x + w, y + h), (0, 255, 255), 2)
                    
                cv2.imshow("Original Video", ImgRec)
            
            if(normalize == True):
                ImgGS = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
                faces = faceDetector.detectMultiScale(ImgGS, 1.3, 5)
                ImgRec = np.copy(Img)
                
                for (x, y, w, h) in faces:
                    roiColor = Img[y:y + h, x:x + w]                
                    cv2.rectangle(ImgRec, (x, y) ,(x + w, y + h), (0, 255, 255), 2)
                    
                cv2.imshow("Original Video", ImgRec)
    
                if(len(faces) > 0):
                    cv2.imshow("Face", roiColor)
    
                    if(nImgs < nTrainImgs):
                        ResultImg = MeanShift(roiColor, ResultImg)
                        print "Trenovaci obrazek: ", nImgs + 1, "z", nTrainImgs
                        nImgs += 1
                        
                    elif(nImgs == nTrainImgs and normalize == True):
                        print "Normalizuji"
                        print "-------------------------------------------------------------------"
                        printHelp()
                        ma = np.max(ResultImg)
                        ResultImg = (ResultImg / ma)
                        normalize = False
                        ResultImg = np.float32(ResultImg)
                        cv2.imshow("ResultImg", ResultImg)
                        
            elif(normalize == False and detect == False):
                cv2.destroyWindow("Clustering Result")
                cv2.destroyWindow("Result with original Value")
                cv2.destroyWindow("Result")
                cv2.destroyWindow("Face")
                
                ImgHSV = cv2.cvtColor(Img, cv2.COLOR_BGR2HSV_FULL)
                bImg = np.zeros((Img.shape), np.float32)
                bImg = ResultImg[ImgHSV[:, :, 1], ImgHSV[:, :, 0]]
                cv2.imshow("Back Projection", bImg)   
                cv2.imshow("Original Video", Img)
                
                if(window == (0, 0, 0, 0)):
                    window = (0, 0, Img.shape[1], Img.shape[0])
                
                ImgRec = np.copy(Img)
                
                #--------------------------------------------------------------
                # Camshift
                ret, window = cv2.CamShift(bImg, window, criteria)
                if(ret):
                    cv2.rectangle(ImgRec, (window[0], window[1]) ,(window[0] + window[2], window[1] + window[3]), (255, 0, 255), 2)
                
                #--------------------------------------------------------------
                # Detekce obliceje
                ImgGS = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
                faces = faceDetector.detectMultiScale(ImgGS, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    roiColor = Img[y:y + h, x:x + w]                
                    cv2.rectangle(ImgRec, (x, y) ,(x + w, y + h), (0, 255, 255), 2)
                #--------------------------------------------------------------
                
                
                cv2.imshow("Original Video", ImgRec)    
                
                    
        key = cv2.waitKey(1)  
        if(key == 27):
            break
        elif(key == ord('r')):# retrain model
            ResultImg = np.zeros((256, 256), np.int)
            nImgs = 0
            normalize = True
            detect = False
    
    cap.release()
    cv2.destroyAllWindows()
  
  
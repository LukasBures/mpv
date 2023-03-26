#MArtin kas / maka
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 14:48:13 2014
"""

import numpy as np
import cv2

#------------------------------------------------------------------------------
# Neni dovoleno vyuzit zadny dalsi import! 
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def CalculateSiftDescriptors(paths, nImg):
    """
    Nacte postupne poporade vsechny obrazky ze vsech trid. Postupne vypocita 
    SIFT deskriptory a pocet descriptoru v jednotlive tride. 
    
    Vrati vektor Descriptors o velikosti X radku (kde X je 125 SIFT deskriptoru
    * 60 trenovacich snimku * 5 trenovacich trid, s tim, ze ve vsech obrazcich 
    neni zaruceno, ze bude detekovano 125 SIFT deskriptoru, tedy muze byt 
    detekovano mene) a 128 sloupcu.
    """
    
    # 125 SIFT deskriptoru.
    sift = cv2.SIFT(125)    
    
    # Vektor 1 * pocet trid, ktery obsahuje na svych pozicich soucet poctu
    # detekovanych SIFT deskriptoru v ramci jedne tridy.
    nDescriptorsInClass = np.zeros((1, len(paths)), np.int)
    
    Descriptors=np.array([[],[]])
    
    for p in range(len(paths)):
        cesta = paths[p]
        n=0
        for i in range(1,nImg+1):
            img = cv2.imread(cesta + "%04d" % i + '.jpg')
            gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            kp,des = sift.detectAndCompute(gray,None)
            n += len(des)
            if p==0 and i==1:
                Descriptors=des.copy()
            else:
                Descriptors=np.vstack((Descriptors,des))
        nDescriptorsInClass[0][p]=n
       
    
    return Descriptors, nDescriptorsInClass


#-----------------------------------------------------------------------------*
def CalculateCentroids(Descriptors, nVisualWord):
    """
    Nashlukuje SIFT 128D deskriptory pomoci metody K-means do K = nVisualWord
    vizualnich trid.
    """
    attempts = 10
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flag = cv2.KMEANS_PP_CENTERS    
    
    # Staci dopsat pouze volani cv2.kmeans(), nebo si ho naprogramovat
    # vlastnimi silami :-) (nedoporucuji). 

    # Vase implementace - jeden radek.
    temp, Label, Center = cv2.kmeans(Descriptors,nVisualWord,criteria,attempts,flag)

    # Label - prirazeni Descriptors do trid.
    # Center - centroidy trid, reprezentujici vizualni slova.
    
    return Label, Center


#-----------------------------------------------------------------------------*
def CalculateModels(Label, nDescriptorsInClass, nVisualWord):
    """
    Vypocet modelu jednotlive tridy. 
    Prumerny vyskyt daneho vizualniho slova v ramci jedne tridy.
    """    
    ModelHistograms = np.zeros((nDescriptorsInClass.shape[1], nVisualWord),
                               np.float64)

    # ModelHistograms = prumerny vyskyt daneho vizualniho slova v ramci jedne
    # tridy.

    for i in range(nDescriptorsInClass.shape[1]):
        for j in range(sum(nDescriptorsInClass[0][0:i]),sum(nDescriptorsInClass[0][0:i+1])):
            ModelHistograms[i][Label.item(j)]+=1            
        ModelHistograms[i][:] = ModelHistograms[i][:]/float(nDescriptorsInClass[0][i])

    
    return ModelHistograms


#-----------------------------------------------------------------------------*
def CalculateExampleHistogram(Img, Centers):
    """
    Klasifikuje SIFT deskriptory (vizualni slova) podle nejblizsiho souseda
    k Centers, vypocte prumerny histogram exampleHistogram vyskytu daneho
    vizualniho slova v obrazku Img.
    """

    sift = cv2.SIFT(125)
    #t=np.append(t,2)
    
    gray= cv2.cvtColor(Img,cv2.COLOR_BGR2GRAY)
    kp,des = sift.detectAndCompute(gray,None)
    exampleHistogram=np.zeros([1,Centers.shape[0]],np.float32)
    t=des.shape[0]
    for i in range(t):
        mdist=np.inf
        w=0
        for j in range(Centers.shape[0]):
            dist=np.linalg.norm(des[i]-Centers[j])
            if dist<mdist:
                mdist=dist
                w=j
        exampleHistogram[0][w] += 1
        
    exampleHistogram = exampleHistogram/float(t)
    # exampleHistogram vypocteny prumerny histogram vyskytu vizualnich slov
    # v ramci jednoho obrazku Img.
    return exampleHistogram


#------------------------------------------------------------------------------
def CalculateAngle(ModelHistograms, exampleHist):
    """
    Vypocita uhel mezi histogramem vzorku a vsemi histogramy z modelu
    a klasifikuje do te tridy s nejmensi hodnotou.
    """
    
    des=[]
    for i in range(ModelHistograms.shape[0]):
        a = np.dot(ModelHistograms[i][:],exampleHist[0])/(np.linalg.norm(ModelHistograms[i][:])*np.linalg.norm(exampleHist[0]))
        print a
        des.append(a)
        
    cls = des.index(max(des))   
    
    
    # cls je cislo tridy, do ktere dany vzorek byl klasifikovan
    return cls


#------------------------------------------------------------------------------
def BoW(Imgs):
    """
    Imgs - vstupni pole BAREVNYCH obrazku (podobny trenovacim).
    """

    # Hlavni funkce, ktera se vola.

    # Poradi odpovida klasifikaci do trid
    # trida 0: drevoTapeta
    # trida 1: kava
    # trida 2: kosik
    # trida 3: zed
    # trida 4: zelezo
    paths = ["./ImgTex/drevoTapeta/", "./ImgTex/kava/", "./ImgTex/kosik/",
             "./ImgTex/zed/", "./ImgTex/zelezo/"]

    # Pocet trenovacich obrazku z kazde tridy.             
    nImg = 60
    
    # Pocet vizualnich slov v BoW metode.
    nVisualWord = 50
    
    # Vypocte deskriptory a jejich pocty v ramci jedne tridy.
    Descriptors, nDescriptorsInClass = CalculateSiftDescriptors(paths, nImg)
    
    # Vypocet vizualnich slov pomoci metody K-means, kde K = nVisualWord.
    Label, Centers = CalculateCentroids(Descriptors, nVisualWord)
    
    # Vypocita modely, ktere jsou reprezentovany histogramy.
    ModelHistograms = CalculateModels(Label, nDescriptorsInClass, nVisualWord)
    
    # Pole vyslednych klasifikaci jednotlivych obrazku.
    cls = np.zeros((len(Imgs), 1), np.int)
    
    # Klasifikace obrazku.
    for i in range(len(Imgs)):
        # Vypocita histogram vstupniho obrazku.
        exampleHist = CalculateExampleHistogram(Imgs[i], Centers)
        
        # Klasifikace.
        cls[i] = CalculateAngle(ModelHistograms, exampleHist)
        
    # Vraci pole klasifikaci.
    print cls
    return cls

##clc;
#import os
#def clc():
#    os.system('cls' if os.name == 'nt' else 'clear')
#    return               
#            
#clc()
#print 'Start'
#
#obr=cv2.imread('./ImgTex/drevoTapeta/0035.jpg')
#
##paths = ["./ImgTex/drevoTapeta/", "./ImgTex/kava/", "./ImgTex/kosik/",
##             "./ImgTex/zed/", "./ImgTex/zelezo/"]
##nImg = 60
##nVisualWord = 50
##Descriptors, nDescriptorsInClass = CalculateSiftDescriptors(paths, nImg)
##Label, Centers = CalculateCentroids(Descriptors, nVisualWord)
##ModelHistograms = CalculateModels(Label, nDescriptorsInClass, nVisualWord)
##exampleHist = CalculateExampleHistogram(obr, Centers)
##cls = CalculateAngle(ModelHistograms, exampleHist)
#print 'Konec'
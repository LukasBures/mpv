import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
print "---START---"

def distance(p1, p2):
    l = min([len(p1),len(p2)])
    dist = 0
    for i in range(l):
        dist += (p1[i]-p2[i])**2
    dist = dist**(1./2.)

    return dist

def countAlfa(hist1, hist2):
    temp = 0
    size1 = 0
    size2 = 0

    for i in range(len(hist1)):
        temp += hist1[i]*hist2[i]
        size1 += hist1[i]**2
        size2 += hist2[i]**2
    size1 = size1**(.5)
    size2 = size2**(.5)
    alfa = np.arccos((temp/(size2*size1)))
    return alfa

def kmeans(descriptors, numOfWords):

    # tato metoda rozedeli vsechny deskriptory do shluku a naleznete klicova slova. Vraci jednak vektor n - klicovych bodu
    # a take list clusteru ke kterym patri deskriptory na jednotlivych pozicich. Jako starovaci vektor slov zvoli meotda prvnich
    # n slov. Zastavovaci podminka je shodnost vektoru klicovych slov ve dvou po sobe jdoucich iteracich

    dimension = 128
    keyWords = np.zeros((numOfWords, dimension))
    for i in range(numOfWords):
        keyWords[i] = descriptors[i]
    numOfIter = 0
    while True:
        labels = np.zeros(len(descriptors))
        # dist = np.zeros((len(descriptors), numOfWords))
        for i in range(len(descriptors)):
            labels[i] = np.argmin(np.sum(np.power(np.subtract(keyWords, descriptors[i]), 2), 1))
        temp = np.zeros((numOfWords, dimension))
        tempCount = 0

        for i in range(len(keyWords)):
            for j in range(len(descriptors)):
                if labels[j] == i:
                    temp[i] += descriptors[j]
                    tempCount += 1
            temp[i] = temp[i]/tempCount
            tempCount = 0
        if (temp == keyWords).all():
            break
        else:
            keyWords = temp
            numOfIter += 1
    return keyWords, labels

def bow(train_data, train_label, test_data, n_visual_words):
    # priprava promennych pro ulozeni vsech deskriptoru a labelu k nim. Kazdy descriptor pak na odpovidajici pozici labelu
    # bude mit informaci o tom, do jake tridy patri
    # objekt SIFT pouzijeme pro detekci deskriptoru a do num of Descriptors ulozime pocet vsech deskriptoru
    allDescriptors = list()
    allLabels = list()

    sift = cv2.xfeatures2d.SIFT_create(125)
    numOfDescriptors = 0

    # promenna idx slouzi k zjisteni poctu ruznych trid
    idx = list()
    # nalezneme deskriptory, ulozime je do jednoho listu
    for i in range(len(train_data)):
        _, d = sift.detectAndCompute(train_data[i], None)
        for j in range(len(d)):
            allDescriptors.append(d[j])
            allLabels.append(train_label[i])
        numOfDescriptors += len(d)
        if not train_label[i] in idx:
            idx.append(train_label[i])
    # print numOfDescriptors


    kWords, allClusters = kmeans(allDescriptors, n_visual_words)
    histograms = np.zeros((len(idx), n_visual_words))
    histogramsCount = np.zeros(len(idx))

    # v teto casti pocitame relativni histogramy
    for i in range(len(allDescriptors)):
        histograms[allLabels[i]][allClusters[i].astype('int')] += 1
        histogramsCount[allLabels[i]] += 1
    for i in range(len(idx)):
        for j in range(n_visual_words):
            histograms[i][j] = histograms[i][j]/histogramsCount[i]

    # inicializace promenne result, ktera v sobe ponese jednotlive labely trid, ke kterym patri testovaci obrazek
    result = np.zeros(len(test_data))

    # print len(test_data)
    # print len(result)

    # nasledujici cast kodu slouzi pro vypocet histogramu testovacich obrazku, naslednemu nalezeni uhlu alfa mezi testovacim
    # histogramem a histogramy testovacich trid. Z nich pak argument minima (index minina) urci k jake tride se testovaci
    # obrazek priradi

    for i in range(len(test_data)):
        _,desc = sift.detectAndCompute(test_data[i], None)
        tempHist = np.zeros(n_visual_words)
        for j in range(len(desc)):
            tempD = np.zeros((n_visual_words))
            for k in range(len(tempD)):
                tempD[k] = distance(desc[j], kWords[k])
            tempHist[np.argmin(tempD)] += 1
        tempHist = tempHist / len(desc)
        histDist = np.empty(len(histograms))
        for j in range(len(histograms)):
            histDist[j] = countAlfa(histograms[j], tempHist)
        result[i] = np.argmin(histDist)
    return result
print "---END--"

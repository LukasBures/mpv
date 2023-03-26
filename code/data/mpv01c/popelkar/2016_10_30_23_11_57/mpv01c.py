from __future__ import division
# -*- coding: utf-8 -*-
#import pylab
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
from math import sqrt
#image = mpimg.imread("000.jpg")

def extend(matrix, pocetNul):

    for i in range(0,pocetNul):

        w, h = len(matrix[0])+2, len(matrix)+2
        extendedMatrix = [[0 for x in range(w)] for y in range(h)]

        for y, val in enumerate(matrix):
            for x, val2 in enumerate(matrix[y]):
                #if y > 0:
                extendedMatrix[y+1][x+1] = matrix[y][x]
        matrix = extendedMatrix
    return extendedMatrix

def convolution(maska, image, jeSobel, pocetNul):
    w, h = len(image[0]) - (pocetNul *2), len(image) - (pocetNul *2)
    gaussMatrix = [[0 for x in range(w)] for y in range(h)]

    if jeSobel:
        nasobek = 1
    else:
        nasobek = 1/len(maska)

    for y, valY in enumerate(image):
        for x, valX in enumerate(image[y]):

            if x < len(image[y]) - (pocetNul*2) and y < len(image) - (pocetNul*2):
                sum = 0
                j = 0
                k = 0

                for i in range(0,len(maska)):

                    if(j == sqrt(len(maska))-1):
                        sum += maska[i] * image[y + k][x + j]
                        #print "if"+str(sum)
                        #print x, y
                        j = 0
                        k += 1
                    else:
                        sum += maska[i] * image[y+k][x + j]
                        #print "else" + str(sum)
                        j += 1
                #print "nas "+str(nasobek)
                #print "sum "+str(sum)
                gaussMatrix[y][x] = int(nasobek *sum)

    return gaussMatrix

def sobel(kernel):

    sobY = [[-1,0,1],[-2, 0, 2],[-1, 0, 1]]
    sobX = [[-1,-2,-1],[0,0,0],[1,2,1]]
    mix = [1, 2, 1, 2, 4, 2, 1, 2, 1]

    if kernel == 3:
        return sobY, sobX
    else:
        while(kernel != len(sobY)):
            sobY = extend(sobY,2)
            sobX = extend(sobX,2)
            sobY = convolution(mix, sobY, 1, 1)
            sobX = convolution(mix, sobX, 1, 1)

    return sobY, sobX

def formatSobel(sobel):
    tmpSobelY = []
    tmpSobelX = []
    for i, val in enumerate(sobel[0]):
        for j, valJ in enumerate(sobel[0][i]):
            tmpSobelY.append(valJ)
    for i, val in enumerate(sobel[1]):
        for j, valJ in enumerate(sobel[1][i]):
            tmpSobelX.append(valJ)
    return tmpSobelY, tmpSobelX

def multiplyMatrix(mat1, mat2):

    w, h = len(mat1[0]), len(mat1)
    matrix = [[0 for x in range(w)] for y in range(h)]

    for i, val in enumerate(mat1):
        for j, val in enumerate(mat1[i]):
            matrix[i][j] = mat2[i][j] * mat1[i][j]
    return matrix

def reflect(matrix, kolikNul):

    for j in range(1,kolikNul+1):

        extended = extend(matrix, 1)

        for i,val in enumerate(extended[2 * j]):
            extended[0][i] = val

        for i, val in enumerate(extended):
            extended[i][0] = extended[i][j*2]
            extended[i][len(extended[0])-1] = extended[i][len(extended[0]) - (2 * j) -1]

        for i, val in enumerate(extended[len(extended) - (2 * j) -1]):
            extended[len(extended)-1][i] = val

        extended[0][0] = extended[2 * j][2 * j]
        extended[0][len(extended[0]) - 1] = extended[2 * j][len(extended[0]) - (2 * j) - 1]

        matrix = extended

    return matrix
        #for k in extended:
        #    print k
        #print ""

def harris(image,maska,sobelSize, k):

    h = pow(maska,2)
    pocetNulBox = int(maska / 2)
    maska = []

    for i in range(0,h):
        maska.append(1)

    pocetNul = int(sobelSize / 2)
    sobelGet = sobel(sobelSize)
    sobelFormatted = formatSobel(sobelGet)

    #image = extend(image, pocetNul)
    image = reflect(image, pocetNul)

    imgSobX = convolution(sobelFormatted[0], image, 0, pocetNul)
    imgSobY = convolution(sobelFormatted[1], image, 0, pocetNul)

    Ixx = multiplyMatrix(imgSobX,imgSobX)
    Iyy = multiplyMatrix(imgSobY,imgSobY)
    Ixy = multiplyMatrix(imgSobY,imgSobX)

    #plt.imshow(Ixx, cmap=pylab.gray())
    #plt.show()

    #plt.imshow(Iyy, cmap=pylab.gray())
    #plt.show()

    #plt.imshow(Ixy, cmap=pylab.gray())
    #plt.show()

    Ixx = extend(Ixx, 1)
    Iyy = extend(Iyy, 1)
    Ixy = extend(Ixy, 1)



    Mxx = convolution(maska, Ixx, 0, pocetNulBox)
    Myy = convolution(maska, Iyy, 0, pocetNulBox)
    Mxy = convolution(maska, Ixy, 0, pocetNulBox)

    w, h = len(Mxx[0]), len(Mxx)
    matrixR = [[0 for x in range(w)] for y in range(h)]

    for i, val in enumerate(Mxx):
        for j, valJ in enumerate(Mxx[i]):
            determinant = (Mxx[i][j] * Myy[i][j])-(Mxy[i][j] * Mxy[i][j])
            trace = Mxx[i][j] + Myy[i][j]
            R = determinant - k*(trace)
            matrixR[i][j] = R

    tmpMin = 100000
    tmpMax = 0

    for i,val in enumerate(matrixR):
        for j, val2 in enumerate(matrixR[i]):
            if val2 < tmpMin:
                tmpMin = val2


    for i,val in enumerate(matrixR):
        for j, val2 in enumerate(matrixR[i]):
            matrixR[i][j] =  matrixR[i][j]+ abs(tmpMin)

    for i,val in enumerate(matrixR):
        for j, val2 in enumerate(matrixR[i]):
            if val2 > tmpMax:
                tmpMax = val2

    for i, val in enumerate(matrixR):
        for j, val2 in enumerate(matrixR[i]):
            matrixR[i][j] = int(round((matrixR[i][j] / tmpMax)*255, 0))

    #plt.imshow(matrixR, cmap=pylab.gray())
    #plt.show()

    return matrixR


#harris(3,3, image, 0.04)



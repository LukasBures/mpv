from __future__ import division
from math import sqrt
import math
# -*- coding: utf-8 -*-
"""
Created on 13:40 6.10.15 

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits: 
@version: 2.0.0
"""

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
                        #print x, y
                        j = 0
                        k += 1
                    else:
                        sum += maska[i] * image[y+k][x + j]
                        j += 1

                gaussMatrix[y][x] = int(round(nasobek *sum, 0))

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

    for i, val in enumerate(mat1):
        for j, val in enumerate(mat1[i]):
            mat1[i][j] = mat2[i][j] * val
    return mat1

def harris(image, maska, sobelSize, k):

    h = pow(maska,2)
    pocetNulBox = int(maska / 2)
    maska = []

    for i in range(0,h):
        maska.append(1)

    pocetNul = int(sobelSize / 2)
    sobelGet = sobel(sobelSize)
    sobelFormatted = formatSobel(sobelGet)

    image = extend(image, pocetNul)

    imgSobY = convolution(sobelFormatted[0], image, 0, pocetNul)
    imgSobX = convolution(sobelFormatted[1], image, 0, pocetNul)

    Ixx = multiplyMatrix(imgSobX,imgSobX)
    Iyy = multiplyMatrix(imgSobY,imgSobY)
    Ixy = multiplyMatrix(imgSobY,imgSobX)

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
            #matrixR[i][j] = (tmp / tmpMax)*255

    for i,val in enumerate(matrixR):
        for j, val2 in enumerate(matrixR[i]):
            if val2 > tmpMax:
                tmpMax = val2

    for i, val in enumerate(matrixR):
        for j, val2 in enumerate(matrixR[i]):
            matrixR[i][j] = round((matrixR[i][j] / tmpMax)*255, 0)

    return matrixR


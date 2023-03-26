# -*- coding: utf-8 -*-
"""
Created on 13:40 6.10.15 

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits: 
@version: 2.0.0
"""
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
import math


def gaussian(ksize):
    mean = ksize/2
    kernel = [[0]*ksize for i in range(ksize)]
    sigma = 1.6
    sm = 0.0
    # print(kernel)
    for i in range(ksize):
        for j in range(ksize):
            kernel[i][j] = math.exp(-0.5 * (((i-mean)/sigma)**2 + ((j-mean)/sigma)**2)) / (2 * math.pi * sigma**2)
            sm += kernel[i][j]

    # print(kernel)
    for i in range(ksize):
        for j in range(ksize):
            kernel[i][j] /= sm
    return kernel

def extend(A, size):
    (xsize, ysize) = get_size(A)
    # print(xsize, ysize)
    # Rozsireni matice o polovinu kernelu kvuli nasobeni
    mat = list()
    for i in range(size):
        mat.append([0] * (ysize + 2 * size))
    for row in A:
        mat.append([0] * size + list(row) + [0] * size)
    for i in range(size):
        mat.append([0] * (ysize + 2 * size))
    return mat


def prod(A, B):
    C = list()
    for i in range(len(A)):
        C.append(list())
        for j in range(len(A[0])):
            C[i].append(A[i][j] * B[i][j])
    return C


def absm(A):
    pom = list()
    for row in A:
        pom.append([abs(x) for x in row])
    return pom


def get_size(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    return rows, cols


def norm(A):
    max_val = max(max(A))+1
    for i in range(len(A)):
        for j in range(len(A[0])):
            A[i][j] = 0 if A[i][j] < 0 else A[i][j] if A[i][j] < 255 else 255#int(A[i][j]/max_val)*255
    return A


def normalize(matrix):
    min_val = min(min(matrix))
    mabs = abs(min_val)

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix[i][j] += mabs

    max_val = max(max(matrix))
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix[i][j] = int((matrix[i][j]/max_val)*255)
    return matrix


def convolution(matrix, kernel):
    """
    Je nutne implementovat a vyuzit - vstupni a vystupni parametry jsou na vas.
    """
    ksize = len(kernel)
    # extd = ksize - 2
    extd = int(ksize/2)
    # print(extd)

    convoluted = [[0] * len(matrix[0]) for x in range(len(matrix))]
    matrix = extend(matrix, extd)
    for i in range(extd, len(matrix)-extd):
        for j in range(extd, len(matrix[0])-extd):

            for q in range(-extd, extd+1):
                for r in range(-extd, extd+1):
                    # print(i-extd, j-extd)
                    convoluted[i-extd][j-extd] += kernel[q + extd][r + extd] * matrix[i + q][j + r]

    # flip kernel
    # pom = list()
    # for row in reversed(kernel):
    #     pom.append([x for x in reversed(row)])
    # kernel = pom
    # print(kernel)
    #
    # (xsize, ysize) = get_size(matrix)
    # print(xsize, ysize)
    # # Rozsireni matice o polovinu kernelu kvuli nasobeni
    # mat = list()
    # for i in range(extd):
    #     mat.append([0]*(ysize+2*extd))
    # for row in matrix:
    #     mat.append([0]*extd + list(row) + [0]*extd)
    # for i in range(extd):
    #     mat.append([0]*(ysize+2*extd))
    #
    # convoluted = [[0 for i in range(ysize)] for j in range(xsize)]
    # # print('convoluted:', convoluted)
    # for i in range(extd, xsize+extd):
    #     for j in range(extd, ysize+extd):
    #         # Compute convolution in all possible x- and y-shifts
    #         for q in range(-extd, extd+1):
    #             for r in range(-extd, extd+1):
    #                 convoluted[i-extd][j-extd] += mat[i+q][j+r] * kernel[q+extd][r+extd]

    # Normalizace
    # for i in range(len(convoluted)):
    #     convoluted[i] = [-x for x in convoluted[i]]     # tohle je přidané !!!!
    # convoluted = normalize(convoluted)
    return convoluted


def blur(img, block_ksize):
    blurMatrix = [[0.0625, 0.125, 0.0625],
                  [0.125, 0.25, 0.125],
                  [0.0625, 0.125, 0.0625]]
    gauss = convolution(img, blurMatrix)
    # gauss = normalize(gauss)
    return gauss



def sobel(img, axis, ksize):
    """
    Je nutne implementovat a vyuzit - vstupni a vystupni parametry jsou na vas.
    """

    K = [[1, 2, 1],
         [2, 4, 2],
         [1, 2, 1]]

    Sx = [[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]]

    Sy = [[1, 2, 1],
          [0, 0, 0],
          [-1, -2, -1]]

    if axis == 'x':
        for i in range(int(ksize / 2) - 1):
            Sx = extend(Sx, 1)
            Sx = convolution(Sx, K)
        return convolution(img, Sx)
    elif axis == 'y':
        for i in range(int(ksize / 2) - 1):
            Sy = extend(Sy, 1)
            Sy = convolution(Sy, K)
        return convolution(img, Sy)
    else:
        raise ValueError("Parameter 'axis' must be 'x' or 'y'!")


def harris(gsimg, block_size, ksize, k):
    """
    Metoda aplikuje Harrisuv detektor vcholu na vstupni obrazek a vrati vysledky ve forme obrazku.

    :param gsimg: Vstupni sedotonovy obrazek.
    :type gsimg: 2D ndarray of uint8

    :param block_size: Velikost kernelu pro rozmazani. Liche cislo >= 3.
    :type block_size: int

    :param ksize: Velikost kernelu Sobelovo operatoru. Liche cislo >= 3.
    :type ksize: int

    :param k: Harrisuv volny parametr v rovnicich.
    :type k: float

    :return corners: Vystupni obrazek s detekovanymi vrcholy (normalizace na hodnoty 0-255)
    :rtype corners: 2D ndarray of uint8
    """

    # w = [[0.0625, 0.125, 0.0625],
    #      [0.125, 0.25, 0.125],
    #      [0.0625, 0.125, 0.0625]]
    w = gaussian(block_size)
    # w = [[1, 1, 1],
    #      [1, 1, 1],
    #      [1, 1, 1]]

    Ix = sobel(gsimg, 'x', ksize)
    # Ix = absm(Ix)
    Iy = sobel(gsimg, 'y', ksize)
    # Iy = absm(Iy)

    # TODO: Compute Ixx and Iyy
    # TODO: Compute convolution G*I(??)

    # Ixy = list()
    # for i in range(len(Ix)):
    #     Ixy.append(list())
    #     for j in range(len(Ix[0])):
    #         Ixy[i].append(Ix[i][j] * Iy[i][j])
    Ixy = prod(Ix, Iy)
    Ixx = prod(Ix, Ix)
    Iyy = prod(Iy, Iy)

    extd = int(block_size / 2)
    Ixx = extend(Ixx, extd)
    Iyy = extend(Iyy, extd)
    Ixy = extend(Ixy, extd)

    corners = [[0]*len(Ix[0]) for i in range(len(Ix))]
    for i in range(extd, len(Ix)):
        for j in range(extd, len(Ix[0])):
            dIxx = 0
            dIyy = 0
            dIxy = 0

            for q in range(-extd, extd+1):
                for r in range(-extd, extd+1):
                    # print(q,r)
                    # convoluted[i-extd][j-extd] += mat[i+q][j+r] * kernel[q+extd][r+extd]
                    dIxx += w[q + extd][r + extd] * Ixx[i + q][j + r]
                    dIyy += w[q + extd][r + extd] * Iyy[i + q][j + r]
                    dIxy += w[q + extd][r + extd] * Ixy[i + q][j + r]

            detM = dIxx*dIyy - dIxy**2
            traceM = dIxx + dIyy
            corners[i][j] = detM - k*traceM**2

    # maxval = 0
    # sigma2 = 4
    # for i in range(len(Ix)):
    #     for j in range(len(Ix[0])):
    #         w = 1#float(2.71**(-(gsimg[i][j]**2 + gsimg[i][j]**2)/(2*sigma2)))
    #         H = [[Ix[i][j]**2, Ix[i][j]*Iy[i][j]],
    #              [Ix[i][j]*Iy[i][j], Iy[i][j]**2]]
    #         # H = np.asarray(H)
    #         # H = np.float64(H)
    #
    #         detM = w*(H[0][0]*H[1][1] - H[1][0]*H[0][1])
    #         traceM = w*(H[0][0] + H[1][1])
    #
    #         R = detM - k*(traceM**2)
    #         print(R)
    #         if R > 10000:
    #             print('Corner found!')
    #             corners.append((i, j))

    corners = normalize(corners)
    return corners

# imgGS = cv2.imread(r"D:\OneDrive\Faculty of Applied Science\Postgraduate Master Studies\2016-2017\MPV\Semester Projects Repository\SP1\c\sampleData\ruka.jpg", cv2.IMREAD_GRAYSCALE)
#
#
# plt.figure(0)
# imgHarris = cv2.cornerHarris(imgGS, 3, 3, 0.04)
# NormHarris = imgHarris + np.abs(imgHarris.min())
# NormHarris = np.int8(255 * (NormHarris / NormHarris.max()))
# plt.imshow(NormHarris, cmap='gray')
#
# plt.figure(1)
# corners = harris(imgGS, 3, 3, 0.04)
# corners = normalize(corners)
# corners = np.asarray(corners)
# NormCorners = corners
# # NormCorners = corners + np.abs(corners.min())
# # NormCorners = 255 * (NormCorners / NormCorners.max())
# # NormCorners = np.int8(NormCorners)
# np.savetxt('numpy_out.txt', NormCorners)
# plt.imshow(NormCorners, cmap='gray')
# plt.show()
#
# print(NormHarris[30])
# print(NormCorners[30])
# print(NormHarris[30]-NormCorners[30])
# print(len(NormHarris), len(NormCorners))
#
# # for i in range(len(corners)):
# #     for j in range(len(corners[0])):
# #         if corners[i][j]-255 > 0:
# #             plt.plot(j,i, 'r*')
# # plt.show()
#
# exit(-8)

# ************************************************************************************************************
# Harris Corner Detector using OpenCV
# imgGS = np.float32(imgGS)
# imgHarris = cv2.cornerHarris(imgGS, 2, 3, 0.04)
#
# cv2.namedWindow("Harris OpenCV", cv2.WINDOW_AUTOSIZE)
#
# # Normalizace
# NormHarris = imgHarris + np.abs(imgHarris.min())
# NormHarris = 255 * (NormHarris / NormHarris.max())
#
# th = 150
# _, ImgThreshold = cv2.threshold(NormHarris, th, 255, cv2.THRESH_BINARY)
#
# cv2.imshow("Harris OpenCV", ImgThreshold)
# cv2.waitKey(0)



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
    sigma = 2.5
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


def extend_image(A, size, type='default'):
    (xsize, ysize) = get_size(A)
    mat = [[0]*(ysize+2*size) for x in range(xsize + 2*size)]

    if type == 'default':
        for i in range(size):
            mat[i][size:-size] = A[0][:]
            mat[len(mat)-1-i][size:-size] = A[-1][:]
        for i in range(size, len(mat)-size):
            mat[i][size:-size] = A[i-size][:]
        for i in range(size):
            for j in range(len(mat)):
                mat[j][i] = mat[j][size]
                mat[j][-1-i] = mat[j][-1-size]
        return mat
    elif type == 'reflect':
        for i in range(size):
            mat[size-1-i][size:-size] = A[i][:]
            mat[len(mat) - 1 - i][size:-size] = A[-size+i][:]
        for i in range(size, len(mat)-size):
            mat[i][size:-size] = A[i-size][:]
        for i in range(size):
            for j in range(len(mat)):
                mat[j][size-1-i] = mat[j][size+i]
                mat[j][-size+i] = mat[j][-1-size-i]
        return mat
    elif type == 'reflect101':
        # Reflect borders on the up and bottom
        i = 1
        pom = 1
        while pom != size+1:
            mat[size - pom][size:-size] = A[abs(i)][:]
            mat[len(mat) - size - 1 + pom][size:-size] = A[-1 - abs(i)][:]
            pom += 1
            i += 1
            if i > len(A) - 1:
                i = -len(A)+2
        # i = 1
        # pom = 1
        # while pom != size + 1:
        #     pom += 1
        #     mat[size-pom+1][size:-size] = A[i][:]
        #     mat[len(mat) - size - 2 + pom][size:-size] = A[-1-i][:]
        #     i += 1
        #     if i > len(A) - 1:
        #         i = 0
        # Copy the former matrix into extended one
        for i in range(size, len(mat)-size):
            mat[i][size:-size] = A[i-size][:]
        # Reflect borders on the left and right
        i = 1
        pom = 1
        while pom != size + 1:
            # print(i)
            for j in range(len(mat)):
                mat[j][size-pom] = mat[j][size + abs(i)]
                mat[j][-size-1+pom] = mat[j][-size-1-abs(i)]
            i += 1
            pom += 1
            if i > len(A) - 1:
                i = -len(A) + 2
        # i = 1
        # pom = 1
        # while pom != size + 1:
        #     print('add')
        #     pom += 1
        #     for j in range(len(mat)):
        #         mat[j][size+1-pom] = mat[j][size + i]
        #         # mat[j][len(mat[0])-size-1+pom] = mat[j][len(mat[0]) - size - i]
        #     i += 1
        #     if i > len(A[0]) - 1:
        #         i = 0
        return mat
    else:
        raise ValueError("Parameter 'type' can be 'default' or 'reflect101' only!")


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
    return poml


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
    min_val = 1000000.0
    for row in matrix:
        for item in row:
            if item < min_val:
                min_val = item
    mabs = abs(min_val)

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix[i][j] += mabs

    max_val = -1000000.0
    for row in matrix:
        for item in row:
            if item > max_val:
                max_val = item

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix[i][j] = int((matrix[i][j]/max_val)*255)
    return matrix


def sobel_kernel(S, K):
    ksize = len(K)
    extd = int(ksize / 2)

    kernel = [[0] * len(S[0]) for x in range(len(S))]
    S = extend(S, 1)
    for i in range(extd, len(S) - extd):
        for j in range(extd, len(S[0]) - extd):

            for q in range(-extd, extd+1):
                for r in range(-extd, extd+1):
                    # print(i-extd, j-extd)
                    kernel[i - extd][j - extd] += K[q + extd][r + extd] * S[i + q][j + r]
    return kernel


def convolution(matrix, kernel):
    """
    Je nutne implementovat a vyuzit - vstupni a vystupni parametry jsou na vas.
    """
    ksize = len(kernel)
    # extd = ksize - 2
    extd = int(ksize/2)
    # print(extd)

    convoluted = [[0] * len(matrix[0]) for x in range(len(matrix))]
    matrix = extend_image(matrix, extd, 'reflect101')
    for i in range(extd, len(matrix)-extd):
        for j in range(extd, len(matrix[0])-extd):

            for q in range(-extd, extd+1):
                for r in range(-extd, extd+1):
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
    #                 convoluted[i-extd][j-extd] += matrix[i+q][j+r] * kernel[q+extd][r+extd]

    # Normalizace
    # for i in range(len(convoluted)):
    #     convoluted[i] = [-x for x in convoluted[i]]     # tohle je přidané !!!!
    # convoluted = normalize(convoluted)
    return convoluted


def transpose(A):
    mat = [[0]*len(A) for i in range(len(A[0]))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            mat[j][i] = A[i][j]
    return mat



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

    Sy = [[-1, -2, -1],
          [0, 0, 0],
          [1, 2, 1]]

    for i in range(int(ksize / 2) - 1):
        # Sx = extend(Sx, 1)
        # Sx = convolution(Sx, K)
        Sx = extend(Sx, 1)
        Sx = sobel_kernel(Sx, K)
    print(Sx)
    # return convolution(img, Sx)
    if axis == 'x':
        # for i in range(int(ksize / 2) - 1):
        #     # Sx = extend(Sx, 1)
        #     # Sx = convolution(Sx, K)
        #     Sx = extend(Sx, 1)
        #     Sx = sobel_kernel(Sx, K)
        # print(Sx)
        return convolution(img, Sx)
    elif axis == 'y':
        # for i in range(int(ksize / 2) - 1):
        #     Sy = extend(Sy, 1)
        #     Sy = convolution(Sy, K)
        #     # Sy = [[-1,-4,-6,-4,-1],
        #     #       [-2,-8,-12,-8,-2],
        #     #       [0,0,0,0,0,],
        #     #       [2,8,12,8,2],
        #     #       [1,4,6,4,1]]
        # print(Sy)
        Sy = transpose(Sx)
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
    # w = gaussian(block_size)
    w = [[float(1)/(block_size**2)]*block_size for i in range(block_size)]

    Ix = sobel(gsimg, 'x', ksize)
    Iy = sobel(gsimg, 'y', ksize)

    Ixy = prod(Ix, Iy)
    Ixx = prod(Ix, Ix)
    Iyy = prod(Iy, Iy)

    extd = int(block_size / 2)
    Ixx = extend_image(Ixx, extd, type='reflect101')
    Iyy = extend_image(Iyy, extd, type='reflect101')
    Ixy = extend_image(Ixy, extd, type='reflect101')

    corners = [[0]*len(Ix[0]) for i in range(len(Ix))]
    for i in range(extd, len(Ix)+extd):
        for j in range(extd, len(Ix[0])+extd):
            dIxx = 0
            dIyy = 0
            dIxy = 0
            for q in range(-extd, extd+1):
                for r in range(-extd, extd+1):
                    dIxx += w[q + extd][r + extd] * Ixx[i + q][j + r]
                    dIyy += w[q + extd][r + extd] * Iyy[i + q][j + r]
                    dIxy += w[q + extd][r + extd] * Ixy[i + q][j + r]
            detM = (dIxx * dIyy) - (dIxy ** 2)
            traceM = dIxx + dIyy
            corners[i-extd][j-extd] = detM - k*(traceM**2)
            # break
        # break
    # for i in range(extd, len(Ix)):
    #     for j in range(extd, len(Ix[0])):
    #         dIxx = 0
    #         dIyy = 0
    #         dIxy = 0
    #
    #         for q in range(-extd, extd+1):
    #             for r in range(-extd, extd+1):
    #                 # print(q,r)
    #                 # convoluted[i-extd][j-extd] += mat[i+q][j+r] * kernel[q+extd][r+extd]
    #                 dIxx += w[q + extd][r + extd] * Ixx[i + q][j + r]
    #                 dIyy += w[q + extd][r + extd] * Iyy[i + q][j + r]
    #                 dIxy += w[q + extd][r + extd] * Ixy[i + q][j + r]
    #
    #         detM = dIxx*dIyy - dIxy**2
    #         traceM = dIxx + dIyy
    #         corners[i][j] = detM - k*(traceM**2)
    return normalize(corners)

# imgGS = cv2.imread(r"D:\OneDrive\Faculty of Applied Science\Postgraduate Master Studies\2016-2017\MPV\Semester Projects Repository\SP1\c\sampleData\rect_micro.jpg", cv2.IMREAD_GRAYSCALE)
#
# plt.figure(0)
# imgHarris = cv2.cornerHarris(imgGS, 5, 3, 0.04, borderType=cv2.BORDER_DEFAULT)
#
# # Normalizace
# NormHarris = imgHarris + np.abs(imgHarris.min())
# NormHarris = np.uint8(255 * (NormHarris / NormHarris.max()))
# # NormHarris = imgHarris
# plt.imshow(NormHarris, cmap='gray')
#
# plt.figure(1)
# corners = harris(imgGS, 5, 3, 0.04)
#
# # corners = np.asarray(corners, dtype=np.float32)
# # NormCorners = corners + np.abs(corners.min())
# # NormCorners = np.uint8(255 * (NormCorners / NormCorners.max()))
# corners = normalize(corners)
# NormCorners = corners
# NormCorners = np.asarray(corners)
# # np.savetxt('numpy_out.txt', NormCorners)
# plt.imshow(NormCorners, cmap='gray')
# plt.show()
#
# print(np.array_equal(NormHarris, NormCorners))
# np.savetxt('numpy_out.txt', NormHarris)
# np.savetxt('numpy_my_out.txt', np.asarray(NormCorners))

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



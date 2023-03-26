# # import cv2
# # import matplotlib.pyplot as plt
# # import pylab
# # image = cv2.imread("000.jpg")
# # from gevent.ares import result
# #
# # image = cv2.imread("000.jpg")
# #
# # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# gausFilter = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
#
# def multiplyMatrix(matrix, matrix2):
#     result = [[0 for x in range(len(matrix[0]))] for y in range(len(matrix))]
#     for i in range(len(matrix)):
#         for j in range(len(matrix[0])):
#             result[i][j] = matrix[i][j]*matrix2[i][j]
#     return result
#
# def extendMatrix(matrix, extension):
#     result = [[0 for x in range(len(matrix)+extension*2)] for y in range(len(matrix)+extension*2)]
#     for i in range(extension, len(result)-extension, 1):
#         for j in range(extension, len(result) - extension, 1):
#             result[i][j] = matrix[i-extension][j-extension]
#     return result
#
# def alternativSobel(size):
#     sobel33 = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
#     sobel = sobel33
#     if size == 3:
#         return sobel
#     else:
#         id = 3
#         while id < size:
#             sobelXX = extendMatrix(sobel, 2)
#             sobel = count_convolution(sobelXX, gausFilter)
#             id += 2
#     return sobel
#
# def countSobel(size):
#     num = 0
#
#     sobel = [[0 for x in range(size)] for y in range(size)]
#     for i in range(0, size//2+1, 1):
#         num += 1
#         for j in range(size//2-1, -1, -1):
#             sobel[i][j] = -(num+(size//2-j)-1)
#             sobel[size-i-1][j] = -(num+(size//2-j)-1)
#             sobel[i][size-j-1] = (num + (size // 2 - j) - 1)
#             sobel[size - i - 1][size-j-1] = ((num + (size // 2 - j) - 1))
#     return sobel
#
# def secondSobel(sobel):
#     size = len(sobel)
#     sobel2 = [[0 for x in range(len(sobel))] for y in range(len(sobel))]
#     for i in range(0, size//2+1, 1):
#         for j in range(0, size//2+1, 1):
#             sobel2[i][j] = sobel[j][i]
#             sobel2[size - i - 1][j] = -sobel[j][i]
#             sobel2[i][size - j - 1] = sobel[j][i]
#             sobel2[size - i - 1][size - j - 1] = -sobel[j][i]
#     return sobel2
#
# def count_convolution(matrix, kernel):
#     result = ([[0 for x in range(len(matrix[0])-len(kernel)//2-1)] for y in range(len(matrix)-len(kernel)//2-1)])
#     for i in range(len(result)):
#         for j in range(len(result[0])):
#             positionij = 0
#             for ii in range(len(kernel)):
#                 for jj in range(len(kernel)):
#                     positionij += kernel[ii][jj] * matrix[i+ii-len(kernel)//2+1][j+jj-len(kernel)//2+1]
#             result[i][j] = positionij
#     return result
#
# def reflect101(matrix, extension):
#     result = [[0 for x in range(len(matrix[0]) + extension * 2)] for y in range(len(matrix) + extension * 2)]
#     for i in range(len(result[0])):
#         for j in range(len(result)):
#             if i < extension and j < extension and i+extension < len(matrix[0]) and j+extension<len(matrix):
#                 x = extension-i
#                 y = extension-j
#             elif i < extension and j > len(matrix):
#                 x = extension - i
#                 y = len(matrix)-(j-len(matrix))
#             elif i > len(matrix[0]) and j < extension:
#                 x = len(matrix[0])-(i-len(matrix[0]))
#                 y = extension-j
#             elif i > len(matrix[0]) and j > len(matrix):
#                 x = len(matrix[0])-(i-len(matrix[0]))
#                 y = len(matrix)-(j-len(matrix))
#             elif i >= extension and (i-extension) < len(matrix[0]) and j < extension:
#                 x = i - extension
#                 y = extension - j
#             elif i < extension and j >= extension and (j-extension) < len(matrix):
#                 x = extension - i
#                 y = j - extension
#             elif i >= extension and i-extension < len(matrix[0]) and j >= extension and j-extension < len(matrix):
#                 x = i - extension
#                 y = j - extension
#             if(x < len(matrix[0]) and y < len(matrix)):
#                 result[j][i] = matrix[y][x]
#     return result
#
# def harris(gsimg, block_size, ksize, k):
#     sobel = alternativSobel(ksize)
#     kernel = [[1 for x in range(block_size)] for y in range(block_size)]
#     extImg = reflect101(gsimg, ksize//2)
#     Ix = count_convolution(extImg, sobel)
#     Iy = count_convolution(extImg, secondSobel(sobel))
#
#     Ixx = multiplyMatrix(Ix, Ix)
#     # plt.imshow(Ixx, cmap=pylab.gray())
#     # plt.show()
#
#     Iyy = multiplyMatrix(Iy, Iy)
#     # plt.imshow(Iyy, cmap=pylab.gray())
#     # plt.show()
#
#     Ixy = multiplyMatrix(Ix, Iy)
#     # plt.imshow(Ixy, cmap=pylab.gray())
#     # plt.show()
#
#     cIxx = count_convolution(reflect101(Ixx, len(kernel)//2), kernel)
#     # plt.imshow(cIxx, cmap=pylab.gray())
#     # plt.show()
#
#     cIyy = count_convolution(reflect101(Iyy, len(kernel)//2), kernel)
#     # plt.imshow(cIyy, cmap=pylab.gray())
#     # plt.show()
#     cIxy = count_convolution(reflect101(Ixy, len(kernel)//2), kernel)
#     # plt.imshow(cIxy, cmap=pylab.gray())
#     # plt.show()
#
#     result = [[0 for x in range(len(gsimg[0]))] for y in range(len(gsimg))]
#     for i in range(len(gsimg)):
#         for j in range(len(gsimg[0])):
#             d = (cIxx[i][j] * cIyy[i][j])-(cIxy[i][j] * cIxy[i][j])
#             t = cIxx[i][j] + cIyy[i][j]
#             R = d - k * t
#             result[i][j] = R
#     # plt.imshow(result, cmap=pylab.gray())
#     # plt.show()
#     #
#     minimum = 1000
#     for i in range(len(result)):
#         for j in range(len(result[0])):
#            if result[i][j] < minimum:
#                minimum = result[i][j]
#     for i in range(len(result)):
#         for j in range(len(result[0])):
#            result[i][j] += minimum
#
#     maximum = 0
#     for i in range(len(result)):
#         for j in range(len(result[0])):
#             if abs(result[i][j]) > maximum:
#                 maximum = abs(result[i][j])
#     for i in range(len(result)):
#         for j in range(len(result[0])):
#             result[i][j] = (result[i][j]/maximum)*255
#
#     # plt.show()
#
#
#     # for i in range(len(result[0])):
#     #     for j in range(len(result)):
#     #         if i-1 > 0 and i+1 < len(result[0]) and j > 0 and j < len(result):
#     #             result[i-1][j-1] = 1 if result[i][j] > 10000 else 0
#     #             result[i+1][j+1] = 1 if result[i][j] > 10000 else 0
#     #             result[i][j] = 1 if result[i][j]>10000 else 0
#     # plt.imshow(result, cmap=pylab.gray())
#     # plt.show()
#     return result
# # harris(image, 3, 3, 0.04)
#
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import pylab
# image = cv2.imread("000.jpg")
# from gevent.ares import result
#
# image = cv2.imread("chessboard.jpg")

# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gausFilter = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]

def multiplyMatrix(matrix, matrix2):
    result = [[0 for x in range(len(matrix[0]))] for y in range(len(matrix))]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            result[i][j] = matrix[i][j]*matrix2[i][j]
    return result

def extendMatrix(matrix, extension):
    result = [[0 for x in range(len(matrix)+extension*2)] for y in range(len(matrix)+extension*2)]
    for i in range(extension, len(result)-extension, 1):
        for j in range(extension, len(result) - extension, 1):
            result[i][j] = matrix[i-extension][j-extension]
    return result

def alternativSobel(size):
    sobel33 = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    sobel = sobel33
    if size == 3:
        return sobel
    else:
        id = 3
        while id < size:
            sobelXX = extendMatrix(sobel, 2)
            sobel = count_convolution(sobelXX, gausFilter)
            id += 2
    return sobel

def countSobel(size):
    num = 0

    sobel = [[0 for x in range(size)] for y in range(size)]
    for i in range(0, size//2+1, 1):
        num += 1
        for j in range(size//2-1, -1, -1):
            sobel[i][j] = -(num+(size//2-j)-1)
            sobel[size-i-1][j] = -(num+(size//2-j)-1)
            sobel[i][size-j-1] = (num + (size // 2 - j) - 1)
            sobel[size - i - 1][size-j-1] = ((num + (size // 2 - j) - 1))
    return sobel

def secondSobel(sobel):
    size = len(sobel)
    sobel2 = [[0 for x in range(len(sobel))] for y in range(len(sobel))]
    for i in range(0, size//2+1, 1):
        for j in range(0, size//2+1, 1):
            sobel2[i][j] = sobel[j][i]
            sobel2[size - i - 1][j] = -sobel[j][i]
            sobel2[i][size - j - 1] = sobel[j][i]
            sobel2[size - i - 1][size - j - 1] = -sobel[j][i]
    return sobel2

def count_convolution(matrix, kernel):
    result = ([[0 for x in range(len(matrix[0])-len(kernel)//2-1)] for y in range(len(matrix)-len(kernel)//2-1)])
    for i in range(len(result)):
        for j in range(len(result[0])):
            positionij = 0
            for ii in range(len(kernel)):
                for jj in range(len(kernel)):
                    positionij += kernel[ii][jj] * matrix[i+ii-len(kernel)//2+1][j+jj-len(kernel)//2+1]
            result[i][j] = positionij
    return result

def reflect101(matrix, extension):
    result = [[0 for x in range(len(matrix[0]) + extension * 2)] for y in range(len(matrix) + extension * 2)]
    result2 = result
    posi = 0
    for i in range(len(result[0])):
        for j in range(len(result)):
            if i < extension and j < extension:
                x = extension-i
                y = extension-j
            elif i>= extension and i < len(matrix[0])+extension and j < extension:
                x = i-extension
                y = extension-j
            elif i >= extension+len(matrix[0]) and j < extension:
                x = (len(matrix[0])-1)-(i-(len(matrix[0])+extension)+1)
                y = extension - j
            elif i < extension and j >=extension and j < len(matrix)+extension:
                x = extension - i
                y = j - extension
            elif i >= extension and i < len(matrix[0])+extension and j >=extension and j < len(matrix)+extension:
                x = i - extension
                y = j - extension
            elif i >= extension + len(matrix[0]) and j >=extension and j < len(matrix)+extension:
                x = (len(matrix[0]) - 1) - (i - (len(matrix[0]) + extension) + 1)
                y = j - extension
            elif i < extension and j >= extension+len(matrix):
                x = extension - i
                y = (len(matrix) - 1) - (j - (len(matrix) + extension) + 1)
            elif i>= extension and i < len(matrix[0])+extension  and j >= extension+len(matrix):
                x = i - extension
                y = (len(matrix) - 1) - (j - (len(matrix) + extension) + 1)
            elif i >= extension+len(matrix[0]) and j >= extension+len(matrix):
                x = (len(matrix[0])-1)-(i-(len(matrix[0])+extension)+1)
                y = (len(matrix) - 1) - (j - (len(matrix) + extension) + 1)

            if (x < len(matrix[0]) and y < len(matrix)):
                result2[j][i] = matrix[y][x]
    return result2

def harris(gsimg, block_size, ksize, k):
    sobel = alternativSobel(ksize)
    kernel = [[1.0/9.0 for x in range(block_size)] for y in range(block_size)]
    extImg = reflect101(gsimg, ksize//2)
    Ix = count_convolution(extImg, sobel)
    Iy = count_convolution(extImg, secondSobel(sobel))

    Ixx = multiplyMatrix(Ix, Ix)
    # plt.imshow(Ixx, cmap=pylab.gray())
    # plt.show()

    Iyy = multiplyMatrix(Iy, Iy)
    # plt.imshow(Iyy, cmap=pylab.gray())
    # plt.show()

    Ixy = multiplyMatrix(Ix, Iy)
    # plt.imshow(Ixy, cmap=pylab.gray())
    # plt.show()

    cIxx = count_convolution(reflect101(Ixx, len(kernel)//2), kernel)
    # plt.imshow(cIxx, cmap=pylab.gray())
    # plt.show()

    cIyy = count_convolution(reflect101(Iyy, len(kernel)//2), kernel)
    # plt.imshow(cIyy, cmap=pylab.gray())
    # plt.show()
    cIxy = count_convolution(reflect101(Ixy, len(kernel)//2), kernel)
    # plt.imshow(cIxy, cmap=pylab.gray())
    # plt.show()


    result = [[0 for x in range(len(gsimg[0]))] for y in range(len(gsimg))]
    for i in range(len(gsimg)):
        for j in range(len(gsimg[0])):
            d = (cIxx[i][j] * cIyy[i][j])-(cIxy[i][j] * cIxy[i][j])
            t = cIxx[i][j] + cIyy[i][j]
            R = d - (k * (t)**2)
            result[i][j] = R
    # plt.imshow(result, cmap=pylab.gray())
    # plt.show()
    minimum = 1000
    for i in range(len(result)):
        for j in range(len(result[0])):
           if result[i][j] < minimum:
               minimum = result[i][j]
    for i in range(len(result)):
        for j in range(len(result[0])):
           result[i][j] += minimum

    maximum = 0
    for i in range(len(result)):
        for j in range(len(result[0])):
            if abs(result[i][j]) > maximum:
                maximum = abs(result[i][j])
    for i in range(len(result)):
        for j in range(len(result[0])):
            result[i][j] = (result[i][j]/maximum)*255

    # plt.show()

    # result = res
    # for i in range(len(result[0])):
    #     for j in range(len(result)):
    #         # if i > 0 and i < len(result[0]) and j > 0 and j < len(result):
    #             # result[i-1][j-1] = 1 if result[i][j] > 10000 else 0
    #             # result[i+1][j+1] = 1 if result[i][j] > 10000 else 0
    #         result[i][j] = 1 if result[i][j] > 10000 else 0
    # plt.imshow(result, cmap=pylab.gray())
    # plt.show()
    # print len(result)
    # print len(result[0])
    # print len(image)
    # print len(image[0])
    return result
# harris(image, 3, 3, 0.04)
# a = np.ones((100,100))
# for i in range(len(a)):
#     for j in range(len(a[0])):
#         a[i][j] = i*10+j
# ref101 =cv2.copyMakeBorder(a,3,3,3,3,cv2.BORDER_REFLECT_101)
# ref102 =  reflect101(a, 3)
# # cv2.imshow("Real Harris",cv2.cornerHarris(image,3,3,0.04))
# print ref101
# print "-------------------------------------"
# for i in range(len(ref102)):
#     print ref102[i]
#
# XXXXX = ref102
# for i in range(len(ref102)):
#     for j in range(len(ref102[0])):
#         if ref102[i][j] == ref101[i][j]:
#             XXXXX[i][j] = 'true'
#         else:
#             XXXXX[i][j] = 'false'
# for i in range(len(ref102)):
#     print ref102[i]
#
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
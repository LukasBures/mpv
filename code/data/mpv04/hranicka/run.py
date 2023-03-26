


import cv2
import numpy as np
import mpv04_ucitel

import mpv04

left = cv2.imread("./data/left.jpg", cv2.IMREAD_GRAYSCALE)
right = cv2.imread("./data/right.jpg", cv2.IMREAD_GRAYSCALE)

f = mpv04_ucitel.find_fundamental_matrix(img_left=left, img_right=right)
print f
print '-------------------'


ff = mpv04.find_fundamental_matrix(img_left=left, img_right=right)
# print ff

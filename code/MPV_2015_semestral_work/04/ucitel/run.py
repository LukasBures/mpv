# -*- coding: utf-8 -*-
"""
Created on 15:39 18.11.15 

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits: 
@version: 1.0.0
"""

import cv2
import pylab
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import mpv04_ucitel


def plot(pt):
    fig = pylab.figure()
    ax = Axes3D(fig)

    ax.scatter(pt[:, 0], pt[:, 1], pt[:, 2])
    ax.set_xlabel('x label')
    ax.set_ylabel('y label')
    ax.set_zlabel('z label')

    pyplot.show()


if __name__ == '__main__':
    left_img = cv2.imread("../data/1.jpg", cv2.IMREAD_GRAYSCALE)
    # left_img = cv2.imread("./cal.png", cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread("../data/0.jpg", cv2.IMREAD_GRAYSCALE)

    pts = mpv04_ucitel.calculate(left_img, right_img)
    plot(pts)
    # cv2.waitKey(0)

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 12:58:47 2014

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits: 
@version: 1.0.0
"""

import cv2
import numpy as np

SZ = 20 # size of each digit is SZ x SZ
PERCENT = 70
DIGITS_FN = 'digits.png'
VERBOSE = False
if __name__ == "__main__":
    print 'splitting ...'
    Img = cv2.imread(DIGITS_FN, 0)
    Train = np.zeros((Img.shape[0], SZ * PERCENT), type(Img[0, 0]))
    Test = np.zeros((Img.shape[0], SZ * (100 - PERCENT)), type(Img[0, 0]))
    
    Train = Img[:, 0:SZ * PERCENT]
    Test = Img[:, (Img.shape[1] - SZ * (100 - PERCENT)):Img.shape[1]]
    
    
    cv2.imwrite("Train.png", Train)
    cv2.imwrite("Test.png", Test)
    
    
    if(VERBOSE):
        cv2.imshow("Img", Img)
        cv2.imshow("Train", Train)
        cv2.imshow("Test", Test)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
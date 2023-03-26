#! /usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
"""
    Dalsi importy 
"""

import numpy as np
import cv2
import mpv04_ucitel as ucitel
"""
    Nacteni SAKoTools
"""

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../"))
import SAKoTools

"""
    Vytvoreni­ parseru parametru
"""
parser = argparse.ArgumentParser(description='Base Validator.')
parser.add_argument('-i', '--input', default=False)
parser.add_argument('-u', '--user', default=False)
parser.add_argument('-t', '--task', default=False)
parser.add_argument('-y', '--year', default=False)
args = parser.parse_args()


def check_rank_fundamental_mat(f):
    rank = np.linalg.matrix_rank(f)
    return rank

user = SAKoTools.getMethodInModule(args.input, 'ucitel', 'find_fundamental_matrix')
path_to_script = os.path.dirname(os.path.abspath(__file__))
left_img = cv2.imread(path_to_script+"/data/left.jpg", cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread(path_to_script+"/data/right.jpg", cv2.IMREAD_GRAYSCALE)

try:
    userResult = user(left_img, right_img)
    teacherResult = ucitel.find_fundamental_matrix(left_img, right_img)
    rank_user = check_rank_fundamental_mat(userResult)
    rank_teacher = check_rank_fundamental_mat(teacherResult)
except Exception as e:
    print str(e)

points = 0

if (userResult.shape[0] == teacherResult.shape[0]) and (userResult.shape[1] == teacherResult.shape[1]):
    if rank_user == rank_teacher == 2:
        if np.allclose(userResult, teacherResult, 1e-8, 1e-8):
            points = 25
else:
    print 'Matice musi mit velikost 3x3'

print 'Za ulohu mate '+ str(points) + ' bodu. Max 25 bodu.'

SAKoTools.saveResults(args.user, args.task, args.year, points, '')

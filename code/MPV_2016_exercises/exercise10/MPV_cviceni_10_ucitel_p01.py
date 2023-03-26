# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 14:47:28 2014

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits: 
@version: 3.0.0

Revision Note:
3.0.0 - 1.12.2016 - Updated for OpenCV 3.1.0 version
"""

import cv2
import numpy as np
import os

if __name__ == '__main__':

    # Ulozit vysledky do souboru?
    VERBOSE = True

    # Cesty ke slozkam.
    # InputDir = "./img/right/"
    InputDir = "./img/left/"
    OutputDir = "./img/output/"

    # Velikost jednoho ctverecku na sachovnici v pozadovane metrice.
    SquareSize = 30.0  # [mm], metrika, na kterou se kalibruje

    # Velikost sachovnice, pocet vrcholu (sloupce, radky).
    PatternSize = (9, 6)

    # Synteticky vygenerovane body.
    PatternPoints = np.zeros((np.prod(PatternSize), 3), np.float32)
    PatternPoints[:, :2] = np.indices(PatternSize).T.reshape(-1, 2)
    PatternPoints *= SquareSize

    ObjPoints = []
    ImgPoints = []
    TermCrit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)

    print "------------------------------------------------------------------"
    print InputDir
    print "------------------------------------------------------------------"

    w = 0
    h = 0
    n = 0
    for fn in os.listdir(InputDir):
        print "Processing %s..." % fn,

        ImgGS = cv2.imread(InputDir + fn, cv2.IMREAD_GRAYSCALE)
        h, w = ImgGS.shape[:2]

        # Nalezne rohy na sachovnici.
        found, corners = cv2.findChessboardCorners(ImgGS, PatternSize)
        if found:
            # Nalezne vrcholy se subpixelovou presnosti.
            cv2.cornerSubPix(ImgGS, corners, (5, 5), (-1, -1), TermCrit)

            # Zapis vysledku do souboru.
            if VERBOSE:
                Vis = cv2.cvtColor(ImgGS, cv2.COLOR_GRAY2BGR)
                cv2.drawChessboardCorners(Vis, PatternSize, corners, found)
                cv2.imwrite(OutputDir + fn, Vis)

        if not found:
            print "chessboard not found"

        # Ulozi synteticke a detekovane body.
        ImgPoints.append(corners.reshape(-1, 2))
        ObjPoints.append(PatternPoints)
        n += 1
        print "OK"

    print "------------------------------------------------------------------"
    rms, cameraMatrix, distCoefs, rvecs, tvecs = cv2.calibrateCamera(ObjPoints,
                                            ImgPoints, (w, h), None, None)
    print "Root-mean-square error:", rms
    print "------------------------------------------------------------------"
    print "Camera matrix:\n", cameraMatrix
    print "------------------------------------------------------------------"
    print "Distortion coefficients: ", distCoefs.ravel()
    print "------------------------------------------------------------------"

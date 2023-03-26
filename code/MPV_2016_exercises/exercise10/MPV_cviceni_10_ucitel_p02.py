# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 17:22:09 2014

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits: 
@version: 3.0.0

Revision Note:
3.0.0 - 1.12.2016 - Updated for OpenCV 3.1.0 version
"""

import cv2

if __name__ == '__main__':
    # SquareSize = 28.0  # [mm], realna metrika, na kterou se kalibruje
    PatternSize = (8, 6)  # pocet vrcholu (sloupce, radky)
    nPt = PatternSize[0] * PatternSize[1]
    TermCrit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE +\
            cv2.CALIB_CB_FAST_CHECK

    print "------------------------------------------------------------------"
    cv2.namedWindow("Camera", 0)
    cap = cv2.VideoCapture(0)

    while True:
        ret, Img = cap.read()

        if ret:
            found, corners = cv2.findChessboardCorners(Img, PatternSize, None, flags)

            if corners is not None:
                print "Corners found:", corners.shape[0], "of", nPt
            else:
                print "Corners found:", str(0), "of", nPt

            if found:
                ImgGS = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
                cv2.cornerSubPix(ImgGS, corners, (5, 5), (-1, -1), TermCrit)
                cv2.drawChessboardCorners(Img, PatternSize, corners, found)

        cv2.imshow("Camera", Img)

        key = cv2.waitKey(1)
        if key == 27:
            print "----------------------------------------------------------"
            print "Terminating ..."
            break

    cap.release()
    cv2.destroyAllWindows()

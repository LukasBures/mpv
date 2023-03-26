# -*- coding: utf-8 -*-
"""
Created on 12:12 6.11.15 

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits: 
@version: 1.0.0
"""

import cv2
import numpy as np

if __name__ == '__main__':

    faceDetector = cv2.CascadeClassifier("./other/haarcascade_frontalface_alt2.xml")

    cap = cv2.VideoCapture(1)
    cv2.namedWindow("Original Video", 1)
    cv2.namedWindow("Face", 1)
    cv2.namedWindow("Original Video ImgRec", 1)

    while True:
        ret, Img = cap.read()
        if ret:
            ImgGS = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
            faces = faceDetector.detectMultiScale(ImgGS, 1.3, 5)
            ImgRec = np.copy(Img)

            for (x, y, w, h) in faces:
                roiColor = Img[y:y + h, x:x + w]
                cv2.rectangle(ImgRec, (x, y), (x + w, y + h), (0, 255, 255), 2)

            if len(faces) > 0:
                cv2.imshow("Face", roiColor)

            cv2.imshow("Original Video ImgRec", ImgRec)
        cv2.imshow("Original Video", Img)
        key = cv2.waitKey(1)

        if key & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()
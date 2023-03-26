# -*- coding: utf-8 -*-
"""
Created on 14:43 2.12.15 

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits: 
@version: 1.0.0
"""

import cv2

cv2.namedWindow("Camera", 0)
cap = cv2.VideoCapture(0)

n = 0
while True:
    ret, Img = cap.read()

    if not ret:
        break

    cv2.imshow("Camera", Img)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('r'):  # 'r' key
        cv2.imwrite("../data/" + str(n) + ".jpg", Img)
        n += 1
    elif key == 27:
        print "Terminating ..."
        break

cap.release()
cv2.destroyAllWindows()

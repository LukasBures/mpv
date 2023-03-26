# -*- coding: utf-8 -*-
"""
Created on Thu Dec 04 15:48:27 2014

@author: Lukas Bures
"""








import cv2




if __name__ == '__main__':
    
    cap = cv2.VideoCapture(0)
    cap.set(3, 1920)# width
    cap.set(4, 1080)# height
    cv2.namedWindow("Image", 0)

    n = 0
    while(True):
        ret, img = cap.read()
        
        if(ret):
            cv2.imshow("Image", img)
        
        key = cv2.waitKey(1)
        if(key == 27):
            break
        elif(key == ord('t')):
            cv2.imwrite(str(n) + ".jpg", img)
            n += 1
            
    cv2.destroyAllWindows()
    cap.release()
            
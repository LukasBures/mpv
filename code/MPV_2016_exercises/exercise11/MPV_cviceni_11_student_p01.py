# -*- coding: utf-8 -*-
"""
Created on Fri Dec 05 10:39:51 2014

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits: 
@version: 2.0.0

Revision Note:
2.0.0 - 5.12.2016 - Updated for OpenCV 3.1.0 version
"""

import cv2
import numpy as np


def draw_matches(img1, kp1, img2, kp2, matches):
    # Create a new output image that concatenates the two images together (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype="uint8")

    # Place the first image to the left
    out[0:rows1, 0:cols1, :] = img1

    # Place the next image to the right of it
    out[0:rows2, cols1:cols1 + cols2, :] = img2

    # For each pair of points we have between both images draw circles, then connect a line between them
    for mat in matches:
        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        B = np.random.randint(0, 256)
        G = np.random.randint(0, 256)
        R = np.random.randint(0, 256)
        
        cv2.circle(out, (int(x1), int(y1)), 3, (B, G, R), 2)   
        cv2.circle(out, (int(x2) + cols1, int(y2)), 3, (B, G, R), 2)

        cv2.line(out, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (B, G, R), 1, cv2.LINE_AA)

    return out


def draw_lines(img, lines, bgr):
    r, c, _ = img.shape
    
    n = 0
    for l in lines:
        l = np.squeeze(l)
        pt1 = np.array([0, l[2] / - l[1]], np.float32)
        pt2 = np.array([c, ((l[0] * c) / - l[1]) + (l[2] / - l[1])], np.float32)
        pt1 = np.int32(pt1)
        pt2 = np.int32(pt2)
        cv2.line(img, (pt1[0], pt1[1]), (pt2[0], pt2[1]), bgr[n], 3, cv2.LINE_AA)
        n += 1
    

if __name__ == '__main__':

    print "Load Images ...",
    Img_l = cv2.imread("./img/l1.jpg", cv2.IMREAD_COLOR)
    Img_r = cv2.imread("./img/r1.jpg", cv2.IMREAD_COLOR)
    print "OK"

    # ------------------------------------------------------------------------------------------------------------------
    # Convert to grayscale images.
    print "Convert to grayscale images ...",

    # Fill yourself.

    print "OK"
    
    # ------------------------------------------------------------------------------------------------------------------
    # Compute SIFT points.
    print "Compute SIFT points ...",
    
    # Fill yourself.
    
    print "OK"
    
    # ------------------------------------------------------------------------------------------------------------------
    # Match SIFT and prepare 2D points.
    print "Match SIFT and prepare 2D points ...",
    
    
    
    # Fill yourself.
    
    
    
    # Apply ratio test.



    # Fill yourself.



    print "OK"
        
    # ------------------------------------------------------------------------------------------------------------------
    # Visualization.
    print "Visualization of keypoints matches ...",






    
    
    # Fill yourself.






    
    
    print "OK"
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # ------------------------------------------------------------------------------------------------------------------
    # Find Fundamental matrix.
    print "Project points in to the second image ...",



    
    # Fill yourself.
    
    
    
    
    nPt = 15
    colors = []
    for i in range(nPt):
        B = np.random.randint(0, 256)
        G = np.random.randint(0, 256)
        R = np.random.randint(0, 256)
        colors.append((B, G, R))
        
        
        
        
        
    # Fill yourself.
        
        
        
        
        
    cv2.namedWindow("Corners l", 0)
    cv2.namedWindow("Corners r", 0)
    cv2.imshow("Corners l", Img_l)
    cv2.imshow("Corners r", Img_r)
    print "OK"
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

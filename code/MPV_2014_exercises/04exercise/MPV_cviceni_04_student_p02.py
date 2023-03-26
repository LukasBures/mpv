# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 23:09:14 2014

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits: Ing. Marek Hruz, Ph.D.
@version: 1.0.0
"""

import cv2
import numpy as np

#-----------------------------------------------------------------------------
# Python implementation of C++ drawMatches() function.
#-----------------------------------------------------------------------------
def drawMatches(img1, kp1, img2, kp2, matches):
    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype = 'uint8')

    # Place the first image to the left
    out[0:rows1, 0:cols1, :] = img1

    # Place the next image to the right of it
    out[0:rows2, cols1:cols1 + cols2, :] = img2

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1), int(y1)), 10, (255, 0, 0), -1)   
        cv2.circle(out, (int(x2) + cols1, int(y2)), 10, (0, 255, 0), -1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1), int(y1)),
                 (int(x2) + cols1, int(y2)), (0, 0, 255), 3)

    return out

#-----------------------------------------------------------------------------
# SIFT, SURF, ORB, detectors, descriptors, matcher, find homography
#-----------------------------------------------------------------------------
if __name__ == '__main__':
    
    # Load color image and convert it in to the grayscale color
    Template = cv2.imread("Stock_84.png", cv2.IMREAD_COLOR)
    TemplateGS = cv2.cvtColor(Template, cv2.COLOR_RGB2GRAY)
    
    cap = cv2.VideoCapture("trimmed.avi")
   
   
   
   
   
   
   
   
   #-------------------------------------------------------------------------
   # Your implementation, SURF, SIFT, ORB, detectors, descriptors, matching, 
   # homography, draw template
   #-------------------------------------------------------------------------
   
   
   
   
   
   
   
   
   
   

    # Main loop
    while(True):

        # Capture frame-by-frame
        ret, Video = cap.read()
        Video = cv2.flip(Video, 0)
        
        # Exit if video ends
        if(ret == False):
            break
        
        
        
        
        
        
        
        
        
        #---------------------------------------------------------------------
        # Your implementation, SURF, SIFT, ORB, detectors, descriptors,
        # matching, homography, draw template
        #---------------------------------------------------------------------











        # Draw results
        cv2.namedWindow("Video")
        cv2.imshow("Video", Video)  

        # ESC key, cv2.waitKey(1) % 0x100 == 27
        if(cv2.waitKey(1) == 27):
            break;

    # Destroy all windows
    cv2.destroyAllWindows()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 12:58:48 2014

@author: Ing. Lukas Bures
@email: lbures@kky.zcu.cz
@credits: 
@version: 1.0.0
"""

import cv2
import numpy as np

import sys





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
        B = np.random.randint(0, 256)
        G = np.random.randint(0, 256)
        R = np.random.randint(0, 256)
        
        cv2.circle(out, (int(x1), int(y1)), 3, (B, G, R), 2)   
        cv2.circle(out, (int(x2) + cols1, int(y2)), 3, (B, G, R), 2)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (B, G, R), 1, cv2.CV_AA)

    return out
    
def drawlines(Img, lines):
    
    r, c, _ = Img.shape
    
    for l in lines:
        l = np.squeeze(l)
        pt1 = np.array([0, l[2] / - l[1]], np.float32)
        pt2 = np.array([c, ((l[0] * c) / - l[1]) + (l[2] / - l[1])], np.float32)
        pt1 = np.int32(pt1)
        pt2 = np.int32(pt2)
        cv2.line(Img, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (0, 128, 0), 1, cv2.CV_AA)

    return Img
    

if __name__ == '__main__':
    VISUALIZATION = True
    VERBOSE = True
    DRAW_EPILINES = True
    
    #--------------------------------------------------------------------------    
    # Load Images
    Img_l = cv2.imread("./img/left/05.jpg", cv2.IMREAD_COLOR)
    Img_r = cv2.imread("./img/right/05.jpg", cv2.IMREAD_COLOR)
    
    if((Img_l == []) or (Img_r == [])):
        print "Nenacetly se obrazky!"
        sys.exit(0)
    
    #--------------------------------------------------------------------------    
    # Convert to grayscale images
    ImgGS_l = cv2.cvtColor(Img_l, cv2.COLOR_RGB2GRAY)
    ImgGS_r = cv2.cvtColor(Img_r, cv2.COLOR_RGB2GRAY)

    #--------------------------------------------------------------------------
    # Compute SIFT points
    sift = cv2.SIFT()
    KP_l, Des_l = sift.detectAndCompute(ImgGS_l, None)
    KP_r, Des_r = sift.detectAndCompute(ImgGS_r, None)
    
    #--------------------------------------------------------------------------
    # Match SIFT and prepare 2D points
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    Matches = matcher.knnMatch(Des_l, Des_r, k = 2)

    # Apply ratio test
    GoodMatches = []
    for m, n in Matches:
        if m.distance < 0.4 * n.distance:
            GoodMatches.append(m)
            
    PT_l = np.float32([KP_l[m.queryIdx].pt for m in GoodMatches]).reshape(-1, 1, 2)
    PT_r = np.float32([KP_r[m.trainIdx].pt for m in GoodMatches]).reshape(-1, 1, 2)
    
    if(VERBOSE):
        print PT_l
        print PT_r

    #--------------------------------------------------------------------------
    # Visualization
    if(VISUALIZATION):
        ImgSIFT_l = cv2.drawKeypoints(ImgGS_l, KP_l[:], None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        ImgSIFT_r = cv2.drawKeypoints(ImgGS_r, KP_r[:], None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        
        
        cv2.namedWindow("SIFT_l")
        cv2.namedWindow("SIFT_r")
        cv2.imshow("SIFT_l", ImgSIFT_l)
        cv2.imshow("SIFT_r", ImgSIFT_r)
        
        matchesImg = drawMatches(Img_l, KP_l, Img_r, KP_r, GoodMatches)
    
        cv2.namedWindow("matchesImg", 0) 
        cv2.imshow("matchesImg", matchesImg)
        cv2.waitKey(0)
    
        cv2.destroyAllWindows()
    #--------------------------------------------------------------------------
    # Najde Fundamentalni matici.
    
    F, mask = cv2.findFundamentalMat(PT_l, PT_r, cv2.FM_LMEDS)

    # We select only inlier points
    PT_l = PT_l[mask.ravel() == 1]
    PT_r = PT_r[mask.ravel() == 1]
    
    if(VERBOSE):
        print
        print "F = ", F
    
    #--------------------------------------------------------------------------
    if(DRAW_EPILINES):
        # Najde body v levo a promitne do prava
        PatternSize = (9, 6) # pocet vrcholu (sloupce, radky)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        TermCrit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)    
    
        found1, corners1 = cv2.findChessboardCorners(Img_l, PatternSize, None, flags)    
        print "Nalezeny vrcholy - levo:", found1
        if(found1):
            Img_lGS = cv2.cvtColor(Img_l, cv2.COLOR_BGR2GRAY)                
            cv2.cornerSubPix(Img_lGS, corners1, (5, 5), (-1, -1), TermCrit)
            cv2.drawChessboardCorners(Img_l, PatternSize, corners1, found1)

        found2, corners2 = cv2.findChessboardCorners(Img_r, PatternSize, None, flags)    
        print "Nalezeny vrcholy - pravo:", found2
        if(found2):
            Img_rGS = cv2.cvtColor(Img_r, cv2.COLOR_BGR2GRAY)                
            cv2.cornerSubPix(Img_rGS, corners2, (5, 5), (-1, -1), TermCrit)
            cv2.drawChessboardCorners(Img_r, PatternSize, corners2, found2)       
       
        #-------------
        lines1 = cv2.computeCorrespondEpilines(corners1, 1, F)
        img5 = drawlines(Img_r, lines1[:9])

        lines2 = cv2.computeCorrespondEpilines(corners2, 2, F)
        img6 = drawlines(Img_l, lines2[:9])
                
        cv2.namedWindow("Corners l", 0) 
        cv2.namedWindow("Corners r", 0) 
        cv2.imshow("Corners l", Img_l)
        cv2.imshow("Corners r", Img_r)
#        cv2.imshow("img5", img5)
#        cv2.imshow("img6", img6)
#        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
            
            
    
    
    
    #--------------------------------------------------------------------------
    # Vypocet projekcnich matic
    P1 = np.zeros((3, 4), np.float)
    P1[0, 0] = P1[1, 1] = P1[2, 2] = 1.0
    
    P2 = np.zeros((3, 4), np.float)
    
    #e2 = np.linalg.solve(F.T, np.zeros((3, 1), np.float))
    _, _, V = np.linalg.svd(F.T, full_matrices = True)
    if(VERBOSE):       
        print "V = ", V
    
    e2 = V[2,:]
   
    if(VERBOSE):    
        print
        print "e2 = ", e2

    u = e2[0]
    v = e2[1]
    w = e2[2]
    S = np.array([[0.0, -w, v], [w, 0.0, -u], [-v, u, 0.0]])
    #S = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    
    if(VERBOSE):
        print
        print "S = ", S
    
    P2[0:3, 0:3] = S*F
    P2[0:3, 3] = e2.T

    if(VERBOSE):
        print
        print "P2 = ", P2
        
        
    #--------------------------------------------------------------------------
    # 3D Rekonstrukce
    pts3d = cv2.triangulatePoints(P1, P2, PT_l, PT_r)
    
    if(VERBOSE):
        print
        print "pts3d = ", pts3d.T
    




































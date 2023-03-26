#!/usr/bin/env python

import numpy as np
import cv2

if __name__ == '__main__':
    imgL = cv2.imread('./tsucuba_left.png')
    imgR = cv2.imread('./tsucuba_right.png')
    gt = cv2.imread('./groundtruth.png', 0)

    print "-------------------------------------------------------------------"
    
    print "Computing disparity ...",
    window_size = 3
    min_disp = 16
    max_disp = 64
    num_disp = max_disp - min_disp

    stereo = cv2.StereoSGBM(minDisparity = min_disp,
                            numDisparities = num_disp,
                            SADWindowSize = window_size,
                            uniquenessRatio = 1,
                            speckleWindowSize = 100,
                            speckleRange = 32,
                            disp12MaxDiff = 1,
                            P1 = 8 * 3 * window_size**2,
                            P2 = 32 * 3 * window_size**2,
                            fullDP = True)
    
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    print "OK"   
    print "-------------------------------------------------------------------"
    
    print "Generating 3d point cloud ...",
    h, w = gt.shape
    f = 0.8 * w #guess for focal length
    Q = np.float32([[1, 0, 0, -0.5 * w],
                    [0,-1, 0,  0.5 * h], # turn points 180 deg around x-axis,
                    [0, 0, 0,       -f], # so that y-axis looks up
                    [0, 0, 1,        0]])
    
    points3D = cv2.reprojectImageTo3D(disp, Q)
    
    Zimg = points3D[:,:,2] - points3D[:,:,2].min()
    Zimg = (Zimg / Zimg.max()) * 255.0
    Zimg = np.uint8(Zimg)        
    
    print "OK"
    print "-------------------------------------------------------------------"

    print "Calculate score ...",            
    gt = np.array(gt, np.float32())
    gt = gt - gt.min()
    gt = (gt / gt.max()) * 255.0# 0.0 - 255.0
    gt = np.uint8(gt)
    
    score = np.sum(np.abs(gt[18:h-18, 18:w-18] - Zimg[18:h-18, 18:w-18]))
    
    print "OK: score =", score
    print "-------------------------------------------------------------------"

    cv2.namedWindow("Side By Side", 0)
    SideBySide = np.hstack((Zimg[18:h-18, 18:w-18], gt[18:h-18, 18:w-18]))
    cv2.imshow("Side By Side", SideBySide)
  
    cv2.namedWindow("R", 0)
    
    cv2.imwrite("Zimg.jpg", Zimg)
    
    
    disp = disp - disp.min()
    disp = (disp / disp.max()) * 255.0
    disp = np.uint8(disp)
    rozdil = np.abs(gt[18:h-18, 18:w-18] - disp[18:h-18, 18:w-18])    
    
    cv2.imshow("R", np.hstack((SideBySide, rozdil)))
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

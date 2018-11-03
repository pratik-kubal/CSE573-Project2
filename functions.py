#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 00:04:34 2018

@author: pratik
"""

def drawlines(img1,img2,lines,pts1,pts2):
    # Reference CV2 Documentation
    # https://docs.opencv.org/3.1.0/da/de9/tutorial_py_epipolar_geometry.html
    import cv2
    import numpy as np
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    colDict = {1:(74, 208, 197),
               2:(140, 20, 143),
               3:(179, 241, 184),
               4:(23, 82, 85),
               5:(135, 0, 55),
               6:(255, 255, 255),
               7:(157, 127, 31),
               8:(198, 167, 158),
               9:(124, 237, 58),
               10:(100, 194, 116)}
    i=1
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = colDict.get(i)
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
        i+=1
    return img1,img2

def normImage(matA):
    import numpy as np
    skeleton=[[] for i in range(0,np.shape(matA)[0])]
    maxValue = 0
    absValue = 0
    for window_h in range(0,np.shape(matA)[0]):
        for window_w in range(0,np.shape(matA)[1]):
            absValue = abs(matA[window_h][window_w])
            skeleton[window_h].append(absValue)     
            if(maxValue < absValue) : maxValue = absValue
    returnMat=[[] for i in range(0,np.shape(matA)[0])]
    for window_h in range(0,np.shape(matA)[0]):
        for window_w in range(0,np.shape(matA)[1]):
            returnMat[window_h].append(skeleton[window_h][window_w] / maxValue)
    return returnMat
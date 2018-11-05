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

def eucl_distance(point,centroid):
    import numpy as np
    return np.sqrt(np.sum(np.square(centroid - point)))

def kmeans(points,centroids,point_centroid_dict):
    import numpy as np
    points = np.matrix(points)
    centroids = np.matrix(centroids)
    i = 0
    for point in points:
        minArray = []
        for centroid in centroids:
            minArray.append(eucl_distance(point,centroid))
        point_centroid_dict.update({i:minArray.index(np.min(minArray))})
        i +=1
    return point_centroid_dict

def updateCentroids(points,centroids,point_centroid_dict):
    import numpy as np
    cluster_centroid_dict={}
    for i,point in enumerate(points):
        cluster = point_centroid_dict.get(i)
        clusterCentroidData=cluster_centroid_dict.get(cluster)
        if(clusterCentroidData is None):
            running_x = point[0]
            running_y = point[1]
            cluster_centroid_dict.update({point_centroid_dict.get(i):[[running_x],[running_y]]})
        else:
           prev_running_x,prev_running_y = clusterCentroidData.copy()
           prev_running_x.append(point[0])
           prev_running_y.append(point[1])
           cluster_centroid_dict.update({point_centroid_dict.get(i):[prev_running_x,prev_running_y]})
    # Recomputing Clusters
    newCentroids = centroids
    for i,_ in enumerate(centroids):
        xcords,ycords = cluster_centroid_dict.get(i)
        newCentroids[i][0] = np.mean(xcords)
        newCentroids[i][1] = np.mean(ycords)
    return newCentroids


def eucl_distance3d(point,centroid):
    # point = [r[0][0],g[0][0],b[0][0]]
    import numpy as np
    return np.sqrt(np.sum(np.square(centroid - point)))


def kmeans3d(raster,centroids,point_centroid_dict):
    import numpy as np
    raster = np.array(raster)
    centroids = np.matrix(centroids)
    for h in range(0,np.shape(raster)[0]):
        rowDict={}
        for w in range(0,np.shape(raster)[1]):
            minArray = []
            for centroid in centroids:
                minArray.append(eucl_distance3d(raster[h][w],centroid))
            rowDict.update({w:minArray.index(np.min(minArray))})
        point_centroid_dict.update({h:rowDict})
    return point_centroid_dict
        
def updateCentroids3d(raster,centroids,point_centroid_dict):
    import numpy as np
    cluster_centroid_dict={}
    for h in range(0,np.shape(raster)[0]):
        for w in range(0,np.shape(raster)[1]):
            cluster = point_centroid_dict.get(h).get(w)
            clusterCentroidData=cluster_centroid_dict.get(cluster)
            if(clusterCentroidData is None):
                running_r = raster[h][w][0]
                running_g = raster[h][w][1]
                running_b = raster[h][w][2]
                cluster_centroid_dict.update({point_centroid_dict.get(h).get(w):[[running_r],[running_g],[running_b]]})
            else:
                prev_running_r,prev_running_g,prev_running_b = clusterCentroidData.copy()
                prev_running_r.append(raster[h][w][0])
                prev_running_g.append(raster[h][w][1])
                prev_running_b.append(raster[h][w][2])
                cluster_centroid_dict.update({point_centroid_dict.get(h).get(w):[prev_running_r,prev_running_g,prev_running_b]})
    newCentroids = np.array(centroids)
    for i,_ in enumerate(centroids):
        if(cluster_centroid_dict.get(i) is not None):
            xcords,ycords,zcords = cluster_centroid_dict.get(i)
            newCentroids[i][0] = np.mean(xcords)
            newCentroids[i][1] = np.mean(ycords)
            newCentroids[i][2] = np.mean(zcords)
    return np.matrix(newCentroids)

def quantaRaster(raster,point_centroid_dict,centroids):
    import numpy as np
    resultImage = []
    for h in range(0,np.shape(raster)[0]):
        holdRow =[]
        for w in range(0,np.shape(raster)[1]):
            r,g,b = np.array(centroids[point_centroid_dict.get(h).get(w)]).flatten()
            holdRow.append([r,g,b])
        resultImage.append(holdRow)
    return resultImage

def eta(data,mu,covar):
    import numpy as np
    left_deno_left = np.sqrt(2*(np.pi))
    left_deno_right = np.sqrt(np.abs(covar))
    left_deno = np.multiply(left_deno_left,left_deno_right)
    left = np.linalg.inv(left_deno)
    designMatrix = np.zeros((len(data),len(mu)))
    for j,point in enumerate(data):
        point = np.transpose(point)
        for i,centroid in enumerate(mu):
            centroid = np.transpose(centroid)
            left = np.subtract(point,centroid)
            left_transpose = np.transpose(left)
            left_transpose = np.dot(left_transpose,np.linalg.inv(covar))
            right = np.dot(left_transpose,left)
            val = np.exp(-0.5*right)
            designMatrix[j][i] = val
    return designMatrix


def gamma(designMat,piMat,nval,kval):
    import numpy as np
    top = piMat[kval]*designMat[nval][kval]
    holdDeno = 0
    for j,pi in enumerate(piMat):
        holdDeno += pi * designMat[nval][j]
    return np.divide(top,holdDeno)

def EStep(designMat,pi):
    import numpy as np
    holdLeft = 0
    holdRight = 0
    for k,_ in enumerate(mu):
        for i,_ in enumerate(X):
            gammaVal = gamma(designMat,pi,i,k)
            left = gammaVal * np.log(pi)
            right = gammaVal * np.log(designMat[i][l])
            holdLeft += left
            holdRight += right
    Q = holdLeft + holdRight
    return Q

def updateMu(data,designMat,mu,piMat):
    import numpy as np
    Nk = []
    for k,_ in enumerate(mu):
        holdval = 0
        for n,_ in enumerate(designMat):
            holdval+=gamma(designMat,piMat,n,kval=k)
        Nk.append(holdval)
        
    newMu = mu.copy()
    for k,mu in enumerate(newMu):
        gammaVal = np.zeros(np.shape(newMu[k]))
        for n,point in enumerate(data):
            temp = np.dot(gamma(designMat,piMat,n,k),point)
            gammaVal = np.add(gammaVal,temp)
        newMu[k] = np.divide(gammaVal,Nk[k])
    return newMu

def MStep():

            
    piMat = []
    for l,_ in enumerate(mu):
        left = 0
        for i,_ in enumerate(designMat):
            left += gamma(designMat,pi,i,l)
        piMat.append(((1/N)*left))
        
    newMu = []
    for l,_ in enumerate(mu):
        holdMuRight = 0
        holdDeno = 0
        for i,_ in enumerate(data):
            holdMuRight += np.multiply(data[i],gamma(designMat,pi,i,l))
            holdDeno += gamma(designMat,pi,i,l)
        val = np.divide(holdMuRight,holdDeno)
        newMu.append(val)
    
    newCovar = []
    for l,centroid in enumerate(mu):
        holdCovar = []
        top = np.zeros((1,1))
        bottom = 0
        for i,_ in enumerate(data):
            gammaVal = gamma(designMat,pi,i,l)
            right = np.square(data[i] - centroid)
            temp = np.multiply(gammaVal,right)
            top = top + temp
            bottom += gammaVal
        holdCovar=np.divide(top,gammaVal)
        newCovar.append(holdCovar)
    
    return piMat,newMu,newCovar
            

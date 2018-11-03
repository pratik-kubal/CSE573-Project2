UBIT = 'pkubal';
import numpy as np;
np.random.seed(sum([ord(c) for c in UBIT]))

import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt
from functions import drawlines

img_left = cv2.imread("./data/tsucuba_left.png")
img_left_g = cv2.cvtColor(img_left.copy(),cv2.COLOR_BGR2GRAY)
img_right = cv2.imread("./data/tsucuba_right.png")
img_right_g = cv2.cvtColor(img_right.copy(),cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
keypoints_img_left = sift.detect(img_left_g.copy(),None)
keypoints_img_right = sift.detect(img_right_g.copy(),None)

display_img1=cv2.drawKeypoints(img_left_g.copy(),keypoints_img_left.copy(),img_left.copy())
display_img2=cv2.drawKeypoints(img_right_g.copy(),keypoints_img_right.copy(),img_right.copy())

cv2.imwrite('2/task2_sift1.jpg',display_img1.copy())
cv2.imwrite('2/task2_sift2.jpg',display_img2.copy())

keypoints_img_leftknn, desc1 = sift.detectAndCompute(img_left_g.copy(),None)
keypoints_img_rightknn, desc2 = sift.detectAndCompute(img_right_g.copy(),None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 50)
search_params = dict(checks=50000)

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(desc1,desc2,k=2)
ratioTestM=[]

for i,(m,n) in enumerate(matches):
    if m.distance < 0.75*n.distance:
        ratioTestM.append([m])
        
img3 = cv2.drawMatchesKnn(img_left_g.copy(),keypoints_img_leftknn.copy(),img_right_g.copy(),keypoints_img_rightknn.copy(),ratioTestM,None,flags =2)

cv2.imwrite('2/task2_matches_knn.jpg',img3)

# Fundamental Matrix
ratioTestM = np.array(ratioTestM)
left_pts = np.float32([keypoints_img_leftknn[m.queryIdx].pt for m in ratioTestM.flatten()])
right_pts = np.float32([keypoints_img_rightknn[m.trainIdx].pt for m in ratioTestM.flatten()])

F, mask = cv2.findFundamentalMat(left_pts,right_pts,cv2.RANSAC,5)
matchesMask = mask.ravel().tolist()
# Random Sampling 10 points
rand10matchesMask = []
inlier_idx = []
for i in range(0,len(matchesMask)):
    if(matchesMask[i] == 1):
        inlier_idx.append(i)
rand10Places = np.random.choice(inlier_idx,10,replace=False)
for i in range(0,len(matchesMask)):
    if((i == rand10Places).any()):
        rand10matchesMask.append(1)
    else:
        rand10matchesMask.append(0)

rand10matchesMask= np.array(rand10matchesMask)

left_pts = left_pts[rand10matchesMask==1]
right_pts = right_pts[rand10matchesMask == 1]

linesonLeft = cv2.computeCorrespondEpilines(right_pts.reshape(-1,1,2), 2,F)
linesonLeft = linesonLeft.reshape(-1,3)
img5,img6 = drawlines(img_left_g,img_right_g,linesonLeft,left_pts,right_pts)

linesonRight = cv2.computeCorrespondEpilines(left_pts.reshape(-1,1,2), 1,F)
linesonRight = linesonRight.reshape(-1,3)
img3,img4 = drawlines(img_right_g,img_left_g,linesonRight,right_pts,left_pts)

cv2.imwrite('2/task2_epi_right.jpg',img5)
cv2.imwrite('2/task2_epi_left.jpg',img3)

# disparity map
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=27)
disparity = stereo.compute(img_left_g,img_right_g)
disparity_norm = normImage(disparity)
disparity_norm = np.asanyarray(np.dot(disparity_norm,255),dtype='int32')
cv2.imwrite('2/task2_disparity.jpg',disparity_norm)
        


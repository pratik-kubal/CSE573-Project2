UBIT = 'pkubal';
import numpy as np;
np.random.seed(sum([ord(c) for c in UBIT]))

import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread("./data/mountain1.jpg")
img1_g = cv2.cvtColor(img1.copy(),cv2.COLOR_BGR2GRAY)
img2 = cv2.imread("./data/mountain2.jpg")
img2_g = cv2.cvtColor(img2.copy(),cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
keypoints_img1 = sift.detect(img1.copy(),None)
keypoints_img2 = sift.detect(img2.copy(),None)

display_img1=cv2.drawKeypoints(img1_g.copy(),keypoints_img1.copy(),img1.copy(),
                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
display_img2=cv2.drawKeypoints(img2_g.copy(),keypoints_img2.copy(),img2.copy(),
                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('1/task1_sift1.jpg',display_img1.copy())
cv2.imwrite('1/task1_sift2.jpg',display_img2.copy())

# Next Part
''' 
Since we have to find best matches of a keypoint in left image on the right 
image, the left image will be the train image while the right will be query 
image
'''
keypoints_img1knn, desc1 = sift.detectAndCompute(img1_g.copy(),None)
keypoints_img2knn, desc2 = sift.detectAndCompute(img2_g.copy(),None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 50)
search_params = dict(checks=50000)

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(desc1,desc2,k=2)
ratioTestM=[]

for i,(m,n) in enumerate(matches):
    if m.distance < 0.75*n.distance:
        ratioTestM.append([m])
        
img3 = cv2.drawMatchesKnn(img1_g.copy(),keypoints_img1knn,img2_g.copy(),keypoints_img2knn,ratioTestM,None,flags =2)

cv2.imwrite('1/task1_matches_knn.jpg',img3)


# Part 3
ratioTestM = np.array(ratioTestM)
src_pts = np.float32([keypoints_img1knn[m.queryIdx].pt for m in ratioTestM.flatten()]).reshape(-1,1,2)
dst_pts = np.float32([keypoints_img2knn[m.trainIdx].pt for m in ratioTestM.flatten()]).reshape(-1,1,2)

M, mask = cv2.findHomography(dst_pts,src_pts, cv2.RANSAC,5)
matchesMask = mask.ravel().tolist()

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

# Random Sampling 10 points
rand10matchesMask = []
inlier_idx = []
for i in range(0,len(matchesMask)):
    if(matchesMask[i] == 1):
        inlier_idx.append(i)
rand10Places = np.random.choice(inlier_idx,10,replace=False)
for i in range(0,len(matchesMask)):
    if((i == rand10Places).any()):
        print(i)
        rand10matchesMask.append(1)
    else:
        rand10matchesMask.append(0)


draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = rand10matchesMask, # draw only inliers
                   flags = 2)

img4 = cv2.drawMatches(img1_g.copy(),keypoints_img1knn,img2_g.copy(),keypoints_img2knn,ratioTestM.flatten(),None,**draw_params)
cv2.imwrite('1/task1_matches.jpg',img4)

# Wraping
translate = [[1,0,500],
             [0,1,700],
             [0,0,1]]
translate = np.matrix(translate)
nM = np.matmul(M,translate)
h,w = np.shape(img2_g)
warpImg = cv2.warpPerspective(img2_g.copy(),M,(w*22,h*22))
#warpImg[0:h,0:w]=img2_g.copy()
cv2.imwrite("1/task1_pano.jpg", warpImg)
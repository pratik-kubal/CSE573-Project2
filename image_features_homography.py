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
        
img3 = cv2.drawMatchesKnn(img1_g.copy(),keypoints_img1knn.copy(),img2_g.copy(),keypoints_img2knn.copy(),ratioTestM,None,flags =2)

cv2.imwrite('1/task1_matches_knn.jpg',img3)


# Part 3
ratioTestM = np.array(ratioTestM)
src_pts = np.float32([keypoints_img1knn[m.queryIdx].pt for m in ratioTestM.flatten()])
dst_pts = np.float32([keypoints_img2knn[m.trainIdx].pt for m in ratioTestM.flatten()])
# ,maxIters=50
H, mask = cv2.findHomography(src_pts,dst_pts, cv2.RANSAC,1,maxIters=2000000000)
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

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = None,
                   matchesMask = rand10matchesMask,
                   flags = 2)

img4 = cv2.drawMatches(img1_g.copy(),keypoints_img1knn,img2_g.copy(),keypoints_img2knn,ratioTestM.flatten(),None,**draw_params)
cv2.imwrite('1/task1_matches.jpg',img4)

# The Homography matrix has some points ofset so need to change that
# https://stackoverflow.com/questions/22220253/cvwarpperspective-only-shows-part-of-warped-image
# Therefore finding real corners after warping
h,w = np.shape(img1_g)
corners = np.array([
  [0,0],
  [0,h-1],
  [w-1,h-1],
  [w-1,0]
])
corners = cv2.perspectiveTransform(np.float32([corners]),H)[0]
bx, by, bwidth, bheight = cv2.boundingRect(corners)
translation = np.array([
  [1,0,-bx],
  [0,1,-by],
  [0,0,1]
])
H2 = np.dot(translation,H)
warpImg = cv2.warpPerspective(img1_g.copy(),H2,((w*2),bheight+15))
warpImg[bheight-h+15:bheight-h+15+h,w:]=img2_g.copy()
#warpImg[0:h,0:w]=img2_g.copy()
cv2.imwrite("1/task1_pano.jpg", warpImg)
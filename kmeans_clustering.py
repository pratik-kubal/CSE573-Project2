UBIT = 'pkubal';
import numpy as np;
import cv2
np.random.seed(sum([ord(c) for c in UBIT]))

import matplotlib.pyplot as plt
from functions import kmeans,updateCentroids,eucl_distance3d,kmeans3d,updateCentroids3d,quantaRaster

X = [[5.9,3.2],
     [4.6,2.9],
     [6.2,2.8],
     [4.7,3.2],
     [5.5,4.2],
     [5,3],
     [4.9,3.1],
     [6.7,3.1],
     [5.1,3.8],
     [6,3]]
centroids = [[6.2,3.2],
             [6.6,3.7],
             [6.5,3]]
colmap = {0:'r',1:'g',2:'b'}

# Computing kmeans
point_centroid_dict = {}
point_centroid_dict = kmeans(X,centroids,point_centroid_dict)

graphX = []
graphY = []

for i, point in enumerate(X):
    graphX.append(np.array(point).flatten()[0])
    graphY.append(np.array(point).flatten()[1])
    color = colmap.get(point_centroid_dict.get(i))
    plt.scatter(graphX[i],graphY[i],marker='^',facecolors=color,edgecolors=color) 
    plt.annotate(point, (graphX[i], graphY[i]))
plt.savefig("3/task3_iter1_a.jpg")

new_centroid = updateCentroids(X,centroids.copy(),point_centroid_dict)
for i in range(0,len(centroids)):
    plt.scatter(*centroids[i], color=colmap[i])
plt.savefig("3/task3_iter1_b.jpg")

point_centroid_dict = kmeans(X,centroids,point_centroid_dict)
new_centroid = updateCentroids(X,centroids.copy(),point_centroid_dict)
plt.close()
for i, point in enumerate(X):
    graphX.append(np.array(point).flatten()[0])
    graphY.append(np.array(point).flatten()[1])
    color = colmap.get(point_centroid_dict.get(i))
    plt.scatter(graphX[i],graphY[i],marker='^',facecolors=color,edgecolors=color) 
    plt.annotate(point, (graphX[i], graphY[i]))
plt.savefig("3/task3_iter2_a.jpg")

new_centroid = updateCentroids(X,centroids.copy(),point_centroid_dict)
for i in range(0,len(centroids)):
    plt.scatter(*centroids[i], color=colmap[i])
plt.savefig("3/task3_iter2_b.jpg")

# Color Quantization
img = cv2.imread("./data/baboon.jpg")

resultImage = []
r,g,b = img[:,:,0],img[:,:,1],img[:,:,2]
for h in range(0,np.shape(img)[0]):
    holdRow =[]
    for w in range(0,np.shape(img)[1]):
        holdRow.append([r[h][w],g[h][w],b[h][w]])
    resultImage.append(holdRow)
resultImage = np.array(resultImage,dtype="float32")
k= 3
centroids=[[np.random.randint(0,255), np.random.randint(0, 255),np.random.randint(0, 255)]
for i in range(k)]
centroids = np.array(centroids,dtype="float64")

old_centroid = np.zeros((np.shape(centroids)))
iteration=1
while(not(np.allclose(old_centroid,centroids,rtol=0.001)) and iteration < 40):
    print(iteration)
    point_centroid_dict = {}
    point_centroid_dict = kmeans3d(resultImage,centroids,point_centroid_dict)
    cold_centroid = centroids
    centroids = updateCentroids3d(resultImage,centroids.copy(),point_centroid_dict)
    iteration+=1
    img2 = quantaRaster(resultImage,point_centroid_dict,centroids)
    display = np.asarray(img2,dtype='float32')
    cv2.imwrite('3/task3_baboon_3.jpg',display)

# k = 5
img = cv2.imread("./data/baboon.jpg")

resultImage = []
r,g,b = img[:,:,0],img[:,:,1],img[:,:,2]
for h in range(0,np.shape(img)[0]):
    holdRow =[]
    for w in range(0,np.shape(img)[1]):
        holdRow.append([r[h][w],g[h][w],b[h][w]])
    resultImage.append(holdRow)
resultImage = np.array(resultImage,dtype="float32")
k= 5
centroids=[[np.random.randint(0,255), np.random.randint(0, 255),np.random.randint(0, 255)]
for i in range(k)]
centroids = np.array(centroids,dtype="float64")
old_centroid = np.zeros((np.shape(centroids)))
iteration=1
while(not (np.allclose(old_centroid,centroids)) and iteration < 40):
    print(iteration)
    point_centroid_dict = {}
    point_centroid_dict = kmeans3d(resultImage,centroids,point_centroid_dict)
    old_centroid = centroids
    centroids = updateCentroids3d(resultImage,centroids.copy(),point_centroid_dict)
    iteration+=1
    
img2 = quantaRaster(resultImage,point_centroid_dict,centroids)
display = np.asarray(img2,dtype='uint8')
cv2.imwrite('3/task3_baboon_5.jpg',display)

# k = 10
img = cv2.imread("./data/baboon.jpg")

resultImage = []
r,g,b = img[:,:,0],img[:,:,1],img[:,:,2]
for h in range(0,np.shape(img)[0]):
    holdRow =[]
    for w in range(0,np.shape(img)[1]):
        holdRow.append([r[h][w],g[h][w],b[h][w]])
    resultImage.append(holdRow)
resultImage = np.array(resultImage,dtype="float32")
k= 10
centroids=[[np.random.randint(0,255), np.random.randint(0, 255),np.random.randint(0, 255)]
for i in range(k)]
centroids = np.array(centroids,dtype="float64")
old_centroid = np.zeros((np.shape(centroids)))
iteration=1
while(not (np.allclose(old_centroid,centroids)) and iteration < 40):
    print(iteration)
    point_centroid_dict = {}
    point_centroid_dict = kmeans3d(resultImage,centroids,point_centroid_dict)
    old_centroid = centroids
    centroids = updateCentroids3d(resultImage,centroids.copy(),point_centroid_dict)
    iteration+=1

img2 = quantaRaster(resultImage,point_centroid_dict,centroids)
display = np.asarray(img2,dtype='uint8')
cv2.imwrite('3/task3_baboon_10.jpg',display)

# k =20
img = cv2.imread("./data/baboon.jpg")

resultImage = []
r,g,b = img[:,:,0],img[:,:,1],img[:,:,2]
for h in range(0,np.shape(img)[0]):
    holdRow =[]
    for w in range(0,np.shape(img)[1]):
        holdRow.append([r[h][w],g[h][w],b[h][w]])
    resultImage.append(holdRow)
resultImage = np.array(resultImage,dtype="float32")
k= 20
centroids=[[np.random.randint(0,255), np.random.randint(0, 255),np.random.randint(0, 255)]
for i in range(k)]
centroids = np.array(centroids,dtype="float64")
old_centroid = np.zeros((np.shape(centroids)))
iteration=1
while(not (np.allclose(old_centroid,centroids)) and iteration < 40):
    print(iteration)
    point_centroid_dict_old = point_centroid_dict.copy()
    point_centroid_dict = {}
    point_centroid_dict = kmeans3d(resultImage,centroids,point_centroid_dict)
    old_centroid = centroids
    centroids = updateCentroids3d(resultImage,centroids.copy(),point_centroid_dict)
    iteration+=1
    img2 = quantaRaster(resultImage,point_centroid_dict,centroids)
    display = np.asarray(img2,dtype='uint8')
    cv2.imwrite('3/task3_baboon_20.jpg',display)

    

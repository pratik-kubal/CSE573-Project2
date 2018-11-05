'''
Color Image Segmentation Using Gaussian Mixture Model and EM Algorithm by Zhaoxia Fu and Liming Wang
Expectation-Maximization. (2018) Color Image Segmentation Using Gaussian Mixture Model and EM Algorithm | SpringerLink. Retrieved November 05, 2018, from https://link.springer.com/chapter/10.1007/978-3-642-35286-7_9
https://link.springer.com/content/pdf/10.1007%2F978-3-642-35286-7_9.pdf
https://link.springer.com/chapter/10.1007/978-3-642-35286-7_9

Updation:
    https://brilliant.org/wiki/gaussian-mixture-model/
    https://en.wikipedia.org/wiki/Mixture_model
    http://statweb.stanford.edu/~tibs/stat315a/LECTURES/em.pdf
    https://stats.stackexchange.com/questions/268895/how-can-i-find-mean-and-covariance-after-em-iteration-on-gmm-algorithmm
    Found out that Christopher Bishop book gives a better formula to calculate covar(sigma)
'''

UBIT = 'pkubal';
import numpy as np;
import cv2
np.random.seed(sum([ord(c) for c in UBIT]))

from functions import eta,EStep,gamma,MStep,updateMu

X = np.matrix([[5.9,3.2],
     [4.6,2.9],
     [6.2,2.8],
     [4.7,3.2],
     [5.5,4.2],
     [5,3],
     [4.9,3.1],
     [6.7,3.1],
     [5.1,3.8],
     [6,3]])
# Each line is a centroid, I have transposed it in the function
mu = np.matrix([[6.2,3.2],
      [6.6,3.7],
      [6.5,3]])

covar = np.matrix([[0.5,0],
         [0,0.5]])

piMat = np.matrix([[1/3],[1/3],[1/3]])

designMat  = eta(X,mu,covar)

newMu = updateMu(X,designMat,mu,piMat)
'''
# E Step 1

Q = EStep(designMat,piMat)

# M Step 1
_,newMu,_ = MStep
        

# New 

'''
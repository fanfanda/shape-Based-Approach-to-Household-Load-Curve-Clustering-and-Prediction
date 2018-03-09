import numpy as np
import math
##from sklearn.metrics.pairwise import pairwise_distances
# from scipy.sparse import lil_matrix
from scipy.spatial.distance import euclidean
from numpy import array, zeros, argmin, inf, equal, ndim
from scipy.spatial.distance import cdist
import sys
def dtw(x, y, length=3, dist=euclidean):
    assert len(x)
    assert len(y)
    if ndim(x)==1:
        x = x.reshape(-1,1)
    if ndim(y)==1:
        y = y.reshape(-1,1)
    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[:,:] = inf
    D1 = D0[1:, 1:]
    D0[0,0] = 0
    for i in range(r):
        for j in range(max(i-length,0),min(i+1+length,c)):
            D1[i,j] = dist(x[i],y[j])
    for i in range(r):
        for j in range(max(i-length,0),min(i+1+length,c)):
            D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])
    print(D1)
    return D1[-1, -1] / sum(D1.shape)
def fastdtw(x, y, dist=euclidean):
    assert len(x)
    assert len(y)
    if ndim(x)==1:
        x = x.reshape(-1,1)
    if ndim(y)==1:
        y = y.reshape(-1,1)
    r, c = len(x), len(y)
    D0 = zeros((r + 3, c + 3))
    D0[:, :] = inf
    D0[2,2] = 0
    D1 = D0[3:, 3:]
    D0[3:,3:] = cdist(x,y,dist)
    D0[3:,3:]=np.square(D1)
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i+2, j+2], D0[i+2, j+1] + D1[i,j-1], D0[i+2, j] + D1[i,j-2] + D1[i,j-1], D0[i+1,j+2] + D1[i-1,j], D0[i,j+2] + D1[i-2,j] + D1[i-1,j])
    return math.sqrt(D1[-1, -1])

a=np.array([1,3,5,7,9])
b=np.array([1,2,1,3,5])
print(fastdtw(a,b))
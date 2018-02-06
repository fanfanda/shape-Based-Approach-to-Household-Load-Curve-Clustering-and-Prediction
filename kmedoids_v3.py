import numpy as np
import math
##from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import euclidean
from numpy import array, zeros, argmin, inf, equal, ndim
from scipy.spatial.distance import cdist
##from dtw import fastdtw
import sys
######################### K-Medoids
def dist(xa,xb):
        return fastdtw(xa, xb,dist=euclidean)
def fastdtw(x, y, dist):
    """
    Computes Dynamic Time Warping (DTW) of two sequences in a faster way.
    Instead of iterating through each element and calculating each distance,
    this uses the cdist function from scipy (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)
    :param array x: N1*M array
    :param array y: N2*M array
    :param string or func dist: distance parameter for cdist. When string is given, cdist uses optimized functions for the distance metrics.
    If a string is passed, the distance function can be 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    if ndim(x)==1:
        x = x.reshape(-1,1)
    if ndim(y)==1:
        y = y.reshape(-1,1)
    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:]
    D0[1:,1:] = cdist(x,y,dist)
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])
    return D1[-1, -1] / sum(D1.shape)

class k_Medoids():
    def __init__(self,data,k=20,batch_size=1000,max_iterators=20):
        self.data=data
        self.k=k
        self.batch_size=batch_size
        self.max_iterators=max_iterators

        #compute dis for each pairs
        print("compute dis for each pairs......")
        self.datalens=len(data)
##        self.pair_dis = pairwise_distances(data, metric=dist)
        self.pair_dis = np.zeros((self.datalens,self.datalens))
        for i in range(self.datalens):
             self.pair_dis[i]=np.nan
             
    def dp_table(self,i,j):
        if math.isnan(self.pair_dis[i][j]):
            self.pair_dis[i][j]=self.pair_dis[j][i]=dist(self.data[i],self.data[j])
        return self.pair_dis[i][j]

    def assign_nearest(self,ids_of_mediods):
##        dists=self.pair_dis[:,ids_of_mediods]
        dists=np.zeros((self.datalens,len(ids_of_mediods)))
        for i in range(self.datalens):
             for j in range(len(ids_of_mediods)):
                  dists[i][j]=self.dp_table(i,ids_of_mediods[j])
        return np.argmin(dists, axis=1)

    def find_medoids(self,assignments,ids_of_medoids):
        medoid_ids = np.full(self.k, -1, dtype=int)
        if self.batch_size:  #is using greedy algorithm ? 0 is not using
            subset = np.random.choice(self.datalens, self.batch_size, replace=False)
        for i in range(self.k):
            if self.batch_size:
                indices = np.union1d(np.intersect1d(np.where(assignments==i)[0], subset),np.array([ids_of_medoids[i]]))
            else:
                indices = np.where(assignments==i)[0]
    ##        distances = dist(x[indices, None, :], x[None, indices, :]).sum(axis=0)
            distances = np.zeros((len(indices),len(indices)))
            
            for k in range(len(indices)):
                 for j in range(len(indices)):
                        distances[k][j]=self.dp_table(indices[k],indices[j])

            distances = distances.sum(axis=0)
            medoid_ids[i] = indices[np.argmin(distances)]
        return medoid_ids

    def kmeds(self):
        print("Initializing to random medoids.")
        ids_of_medoids = np.random.choice(self.datalens, self.k, replace=False)
        class_assignments = self.assign_nearest(ids_of_medoids)

        for i in range(self.max_iterators):
            print("\tFinding new medoids.")
            ids_of_medoids = self.find_medoids(class_assignments,ids_of_medoids)
            print("\tReassigning points.")
            new_class_assignments = self.assign_nearest(ids_of_medoids)

            diffs = np.mean(new_class_assignments != class_assignments)
            class_assignments = new_class_assignments

            print("iteration {:2d}: {:.2%} of points got reassigned."
                  "".format(i, diffs))
            if diffs <= 0.01:
                break
        return class_assignments, ids_of_medoids

# ## Generate Fake Data
# print("Initializing Data.")
# ds = 5
# ks = 6
# ns = ks * 10000
# #generate test data......
# data = np.random.normal(size=(ns, ds))
# for kk in range(ks):
#     dd = (kk-1)%ds
#     data[kk*ns//ks:(kk+1)*ns//ks,dd] += 3*ds*kk
# ###compute dis for each pairs
# ##print("compute dis for each pairs......")
# ##pair_dis = pairwise_distances(x, metric=dist)

# ## doing the k-medoids clustering....
# print("doing Kmedoids.....")
# t = k_Medoids(data)
# final_assignments, final_medoid_ids = t.kmeds()


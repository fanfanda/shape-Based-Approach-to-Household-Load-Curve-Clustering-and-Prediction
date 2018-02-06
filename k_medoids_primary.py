import numpy as np
from dtw import dtw
from scipy.spatial.distance import euclidean
import sys
######################### K-Medoids
def dist(xa,xb):
        return np.sqrt(np.sum(np.square(xa-xb), axis=-1))
##        return np.array(map(lambda x,y:dtw(x, y, dist=euclidean),xa,xb))
class k_Medoids():
    def __init__(self,data,k=12,batch_size=1000,max_iterators=20):
        self.data=data
        self.k=k
        self.batch_size=batch_size
        self.max_iterators=max_iterators

        #compute dis for each pairs
        print("compute dis for each pairs......")
        self.datalens=len(data)
        
    def assign_nearest(self,ids_of_mediods):
        dists = dist(self.data[:,None,:], self.data[None,ids_of_mediods,:])
        return np.argmin(dists, axis=1)

    def find_medoids(self,assignments,ids_of_medoids):
        medoid_ids = np.full(self.k, -1, dtype=int)
        if self.batch_size:  #is using greedy algorithm ? 0 is not using
            subset = np.random.choice(self.datalens, self.batch_size, replace=False)
        for i in range(self.k):
            if self.batch_size:
                indices = np.union1d(np.intersect1d(np.where(assignments==i)[0], subset),ids_of_medoids[i])
            else:
                indices = np.where(assignments==i)[0]
    ##        distances = dist(x[indices, None, :], x[None, indices, :]).sum(axis=0)

            distances = dist(self.data[indices, None, :], self.data[None, indices, :]).sum(axis=0)
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
# ds = 3
# ks = 12
# ns = ks * 10000
# #generate test data......
# data = np.random.normal(size=(ns, ds))
# for kk in range(ks):
#     dd = (kk-1)%ds
#     data[kk*ns//ks:(kk+1)*ns//ks,dd] += 3*ds*kk
# ##compute dis for each pairs
# ##print("compute dis for each pairs......")
# ##pair_dis = pairwise_distances(x, metric=dist)

# ## doing the k-medoids clustering....
# print("doing Kmedoids.....")
# t = k_Medoids(data)
# final_assignments, final_medoid_ids = t.kmeds()


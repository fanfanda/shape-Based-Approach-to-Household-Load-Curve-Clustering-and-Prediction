# -*- coding: utf-8-*- 
from user_electric import *
import os
from kmedoids_v3 import *
from S_Dbw import *
user_data = []
count = 0
exit_flag = False
data_size = 80000
cluster_num = 200
#read data from database
print("read data from database....")
for i in range(50):
    with open('/home/fanfanda/portData2015/part-000'+str(i).zfill(2), 'r') as f:                          
         data = f.readlines()  #txt中所有字符串读入data
         for index, item in enumerate(data):
             count+=1
             if count==data_size:
                 exit_flag=True
                 break
             meta_data=user_electric(item.rstrip('\n').split(','))
             if sum(meta_data.electric_data)==0.0:
                 continue
             user_data.append(meta_data)
         if exit_flag:
            break
electric_data=np.zeros(shape=(len(user_data),24))
for index,item in enumerate(user_data):
    electric_data[index]=item.reduce_normalized_electric_data
if not os.path.exists("data_"+str(data_size)+".npy"):
    print("saving data.....")
    np.save("data_"+str(data_size)+".npy",electric_data)

print("doing the cluster....")
#doing the cluster
cluster = k_Medoids(data=electric_data,k=cluster_num,batch_size=1000)
final_assignments, final_medoid_ids = cluster.kmeds()

if not os.path.exists("final_assignments_"+str(data_size)+"_"+str(cluster_num)+".npy"):
    np.save("final_assignments_"+str(data_size)+"_"+str(cluster_num)+".npy",final_assignments)
if not os.path.exists("final_medoid_ids_"+str(data_size)+"_"+str(cluster_num)+".npy"):
    np.save("final_medoid_ids_"+str(data_size)+"_"+str(cluster_num)+".npy",final_medoid_ids)

S_Dbwresult = S_Dbw(electric_data,final_assignments,final_medoid_ids)
print(S_Dbwresult.S_Dbw_result())
         






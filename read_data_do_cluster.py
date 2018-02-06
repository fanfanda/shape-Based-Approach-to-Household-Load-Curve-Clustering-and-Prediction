# -*- coding: utf-8-*- 
from user_electric import *
from k_medoids_primary import *
user_data=[]
count=0
#read data from database
print("read data from database....")
for i in range(50):
    with open('/home/fanfanda/portData2015/part-000'+str(i).zfill(2), 'r') as f:                          
         data = f.readlines()  #txt中所有字符串读入data
         for index, item in enumerate(data):
             meta_data=user_electric(item.rstrip('\n').split(','))
             if sum(meta_data.electric_data)==0.0:
                 continue
             user_data.append(meta_data)
electric_data=np.zeros(shape=(len(user_data),96))
for index,item in enumerate(user_data):
    electric_data[index]=item.normalized_electric_data
print("doing the cluster....")
#doing the cluster
cluster = k_Medoids(data=electric_data,k=20,batch_size=1000)
final_assignments, final_medoid_ids = cluster.kmeds()

final_assignments=np.array(final_assignments)
np.savetxt("cluster_result.txt", final_assignments)
# file=open('cluster_result.txt','w')
# file.write(str(final_assignments))
# file.close()

file=open('cluster_medoids.txt','w')
file.write(str(final_medoid_ids))
file.close()

         
         



# a=np.array([1,2,3,4,5,6,7,8,9])
# t=user_electric.user_electric(a)





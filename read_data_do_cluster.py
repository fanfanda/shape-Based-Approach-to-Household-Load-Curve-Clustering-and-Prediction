# -*- coding: utf-8-*- 
from user_electric import *
# from k_medoids_primary import *
from kmedoids_v3 import *
user_data=[]
count=0
exit_flag=False
#read data from database
print("read data from database....")
for i in range(50):
    with open('/home/fanfanda/portData2015/part-000'+str(i).zfill(2), 'r') as f:                          
         data = f.readlines()  #txt中所有字符串读入data
         for index, item in enumerate(data):
             count+=1
             if count==200:
                 exit_flag=True
                 break
             meta_data=user_electric(item.rstrip('\n').split(','))
             if sum(meta_data.electric_data)==0.0:
                 continue
             user_data.append(meta_data)
         if exit_flag:
            break
electric_data=np.zeros(shape=(len(user_data),96))
for index,item in enumerate(user_data):
    electric_data[index]=item.reduce_normalized_electric_data
print("doing the cluster....")
#doing the cluster
cluster = k_Medoids(data=electric_data,k=5,batch_size=100)
final_assignments, final_medoid_ids = cluster.kmeds()

file=open('cluster_result.txt','w')
for i in final_assignments:
    file.write(str(i)+'\n')
file.close()

file=open('cluster_medoids.txt','w')
file.write(str(final_medoid_ids))
file.close()

         
         



# a=np.array([1,2,3,4,5,6,7,8,9])
# t=user_electric.user_electric(a)





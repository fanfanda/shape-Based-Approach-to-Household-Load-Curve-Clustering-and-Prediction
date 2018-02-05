# -*- coding: utf-8-*- 
from user_electric import *

user_data=[]
#read data from database
for i in range(50):
    with open('/home/fanfanda/portData2015/part-000'+str(i).zfill(2), 'r') as f:                          
         data = f.readlines()  #txt中所有字符串读入data
         for index, item in enumerate(data):
             print(item)
             meta_data=user_electric(item.rstrip('\n').split(','))
             user_data.append(meta_data)

print(user_data[0].normalized_electric_data)
print(len(user_data))
         
         
         



# a=np.array([1,2,3,4,5,6,7,8,9])
# t=user_electric.user_electric(a)





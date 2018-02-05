# -*- coding: utf-8-*- 
import user_electric
import numpy as np

for i in range(50):
    with open('/home/fanfanda/portData2015/part-000'+str(i).zfill(2), 'r') as f:                          
         data = f.readlines()  #txt中所有字符串读入data
         print(data)
         print(type(data))
         break
         
         



# a=np.array([1,2,3,4,5,6,7,8,9])
# t=user_electric.user_electric(a)





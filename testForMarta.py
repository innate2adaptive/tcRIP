# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 11:34:39 2017

@author: lewismoffat
"""
import matplotlib.pyplot as plt
import numpy as np

# used numpy version as its faster and easier to read
pink_x = np.arange(0, 7501, dtype=float)
pink_y = np.zeros_like(pink_x) # this makes an array of zeros the same shape as pink_x
pink_y[:]=2 # make all the values in the vector now equal 2
 

          
          
for idx, value in enumerate(pink_x): # this will go value by value and we can 
# use the index for changing the corresponding y values
    if value in [748, 1198, 1393, 1813]: # if the x value is in here
        pink_y[idx]=2.5 # then change its corresponding y value to 2.5 instead of the default 2
          



plt.plot(pink_x, pink_y, marker='o')
plt.title("SNPs")

plt.show()
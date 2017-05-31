# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 15:00:23 2017

Takes the arrays of sequences and then concats them and calculates their 
lengths with the aim of creating a historgram of lengths.

@author: lewismoffat
"""

import numpy as np
import matplotlib.pyplot as plt

# remove the count columns from each array
cd4a=np.delete(np.load('data/cd4A.npy'),1,1)
cd8a=np.delete(np.load('data/cd8A.npy'),1,1)
cd4b=np.delete(np.load('data/cd4B.npy'),1,1)
cd8b=np.delete(np.load('data/cd8B.npy'),1,1)
# each vector is ['seq']xnumber of sequences as np.array

# combine the vectors into one big column vector 
cd=np.concatenate((cd8a,cd4a,cd8b,cd4b),axis=0)

# free up memory. Dont ue del as it can leave ghost variables
cd4a=None
cd8a=None
cd4b=None
cd8b=None


lenArr=[]
# create new vector of lengths 
for seq in cd:
    lenArr.append(len(seq[0]))
    
# convert list to numpy array
lenArr=np.asarray(lenArr)

# free up memory
cd = None
seq = None

# bins
binNum=max(lenArr)-min(lenArr)
bins=np.arange(min(lenArr),max(lenArr))
plt.hist(lenArr, bins=binNum)
plt.xticks(bins)
plt.title("Sequence Length Histogram")
plt.xlabel("Sequence Length (AA)")
plt.ylabel("Frequency")
plt.show()

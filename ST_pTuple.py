# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 10:56:51 2017

@author: lewismoffat

This script is focused on statistics, it calculates the most common pTuples
without clipping and with clipping

"""

#==============================================================================
# Module Imports 
#==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import dataProcessing as dp
import pdb
import seaborn as sns 

from collections import defaultdict
from collections import Counter

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from matplotlib import pylab

#==============================================================================
# Get the data
#==============================================================================

# are we doing the full set
singlePatient=False



# which patient to get data from 
patient=['Complete']
chain = "beta"


if singlePatient:
    print('Patient: '+patient[0])
    delim = ["naive",chain]+patient #other delimiters
else:
    print('Patient: All')
    delim = ["naive",chain] #other delimiters


seqs, vj = dp.loadAllPatients(delim) # these gets all the sequences and vj values



#==============================================================================
# Clipping the data
#==============================================================================

# filter out joint sequences
seqs[0], seqs[1], vj[0], vj[1], joint = dp.removeDup(seqs[0], seqs[1], vj[0], vj[1])

# filter to 14 long, together its still 200k seqs
seqs[0]=dp.filtr(seqs[0], 14)
seqs[1]=dp.filtr(seqs[1], 14)


# clip the sequences
for idx, group in enumerate(seqs):
    for idx2, seq in enumerate(group):
        group[idx2]=seq[4:10]

# get tuples, list is already flat  
seqs[0]=dp.expandTuples(seqs[0],n=4)
seqs[1]=dp.expandTuples(seqs[1],n=4)

# make counters
c4=Counter(seqs[0])
c8=Counter(seqs[1])

print(c4.most_common(n=10))
print()
print(c8.most_common(n=10))














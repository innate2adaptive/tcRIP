# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 11:58:21 2017

@author: lewismoffat


This script runs an AutoEncoder netowork on all the data for a specified Feat.
Engineering methods

"""

#==============================================================================
# Module Imports 
#==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import dataProcessing as dp
import pdb
import seaborn as sns 
import sklearn as sk

from collections import defaultdict
from collections import Counter

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder

from matplotlib import pylab

from hmmlearn import hmm

#==============================================================================
# Get the data
#==============================================================================

method = "Li" #"tuple"
tupsize=3

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

# filter out joint sequences
seqs[0], seqs[1], vj[0], vj[1], joint = dp.removeDup(seqs[0], seqs[1], vj[0], vj[1])


#integerize 
seqs[0]=dp.char2int(seqs[0], 14, pad=False)
seqs[1]=dp.char2int(seqs[1], 14, pad=False)

for idx0,seqz in enumerate(seqs):
    for idx, seq in enumerate(seqz):
        seqs[idx0][idx].append(21)
cd4t,cd4v=train_test_split(seqs[0], test_size=0.2)
cd8t,cd8v=train_test_split(seqs[1], test_size=0.2)

# need to make 21 the end token

#==============================================================================
#  Calculate transition matrix
#==============================================================================


az=cd4t
b = np.zeros((21,21)) # empty matrix to be filled

for a in az: # go through each sequence
    for (x,y), c in Counter(zip(a, a[1:])).items(): # go through each counter a
        b[x-1,y-1] += c
# normalize the rows
row_sums = b.sum(axis=1)
new_matrix = b / row_sums[:, np.newaxis]


print(b)



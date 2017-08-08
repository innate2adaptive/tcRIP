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

cd4t,cd4v=train_test_split(seqs[0], test_size=0.2)
cd8t,cd8v=train_test_split(seqs[1], test_size=0.2)

# format test set
xtest, ytest = dp.dataCreator(cd4t,cd8t)
#xtest=xtest.tolist()

xtest, ytest= sk.utils.shuffle(xtest,ytest)



#==============================================================================
# Training two HMMS, one for each class
#==============================================================================
#CD4 sequences
lenArr=[]
for idx, seq in enumerate(cd4t):
    lenArr.append(len(seq)) # get the length
lenArr=np.array(lenArr)        


# init the HMM
hmm4=hmm.MultinomialHMM(n_components=20,
                        random_state=42,
                        verbose=1)

# fit with Baum-Welch


lb=LabelEncoder().fit([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
for idx, seq in enumerate(cd4t):
    seq=lb.transform(seq)
    seq=np.atleast_2d(seq).T
    cd4t[idx]=seq

pdb.set_trace()



hmm4.fit(np.atleast_2d(cd4t).T,lenArr)






#CD84 sequences
lenArr=[]
for idx, seq in enumerate(cd4t):
        lenArr.append(len(seq)) # get the length
        cd8t[idx]=list(seq)     # convert to get the characters


# init the HMM
hmm8=hmm.MultinomialHMM(n_components=20,
                        random_state=42,
                        verbose=1)
# fit with Baum-Welch
hmm8.fit(cd8t,lenArr)

# get the posterior prob of val set
y_pred4=hmm4.predict_proba(xtest)
y_pred8=hmm8.predict_proba(xtest)






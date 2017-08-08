# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 11:58:21 2017

@author: lewismoffat


This script runs an AutoEncoder network on all the data for a specified Feat.
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

from collections import defaultdict
from collections import Counter

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split 

from matplotlib import pylab

import autoencoderModel as ae

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

#==============================================================================
# Feature Engineering
#==============================================================================

if method=="Li":
    seqs[0]=dp.filtr(seqs[0], 14)
    seqs[1]=dp.filtr(seqs[1], 14)
    
    seqs[0]=dp.seq2fatch(seqs[0])
    seqs[1]=dp.seq2fatch(seqs[1])
    
    # use function to create data
    X, y = dp.dataCreator(seqs[0],seqs[1])
else:
    X=dp.char2ptuple(seqs[0]+seqs[1], n=tupsize)
    # use function to create data
    X, y = dp.dataCreator(seqs[0],seqs[1])
    
#==============================================================================
# Model Spec. 
#==============================================================================
    

aeControllerParams={'batch_size':32,
                     'epochs':500,
                     'learningRate':1}
                     
ModelParams={'learningRate':0.1,
                'embedding_size':10,
                'vocab_size':22,
                'cell_size':128,
                'LSTM':False, # if false it uses GRU
                'stacked':False,
                'dropout':True,
                'unidirectional':True,
                'attention':True,
                'atten_len':2,
                'regu':False, # not being used
                'batch_norm':False, # uses batch norm on the last affine layers
                'onlyLinear':False,
                'conv':False,
                'embed':False,
                'save':True,
                'load':False}
                
                
            
ModelParams['maxLen']=X.shape[1]
                
xTrain, xVal, yTrain, yVal = train_test_split(X, y, test_size=0.20) 


# Spool up warp drives! This gets the rnn controller class going
aeMain = ae.Controller(aeControllerParams, ModelParams) 
print("Training AE")
xTrain, xVal = aeMain.train(xTrain, yTrain, xVal, yVal)




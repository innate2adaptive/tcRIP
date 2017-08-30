# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 11:58:21 2017

@author: lewismoffat


This script runs an AutoEncoder network on all the data for a specified Feat.
Engineering methods. 

"""

#==============================================================================
# Module Imports 
#==============================================================================

import numpy as np
import dataProcessing as dp
import pdb
import sklearn as sk

from sklearn.model_selection import train_test_split 
from scipy.sparse import hstack

import UL_autoencoderModel as ae

#==============================================================================
# Get the data
#==============================================================================

method ="tuple" #"Li" #
loadCDR12=False
tupsize= 2
length = 14
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
# Helper Functions
#==============================================================================
def createCDR(seqs,length):
    c1=[]
    c2=[]
    c3=[]
    for idx, seq in enumerate(seqs):
        c1.append(seq[:length])
        c2.append(seq[length:(length+6)])
        c3.append(seq[(length+6):])
    return c1,c2,c3


#==============================================================================
# Feature Engineering
#==============================================================================

if method=="Li":
    # filter the sequnces
    seqs[0]=dp.filtr(seqs[0], 14)
    seqs[1]=dp.filtr(seqs[1], 14)
    
    
    
    if loadCDR12:
        # add extrac cds
        cddict=dp.extraCDs()
        seqs[0], vj[0]=dp.addCDextra(seqs[0],vj[0],cddict)
        seqs[1], vj[1]=dp.addCDextra(seqs[1],vj[1],cddict)
        seqs[0]=dp.filtr(seqs[0], 14+5+6)
        seqs[1]=dp.filtr(seqs[1], 14+5+6)
    
    seqs[0]=dp.seq2fatch(seqs[0])
    seqs[1]=dp.seq2fatch(seqs[1])
    
    # use function to create data
    X, y = dp.dataCreator(seqs[0],seqs[1])
else:
    
    if loadCDR12:
        length=length+5+6
        # add extrac cds
        cddict=dp.extraCDs()
        seqs[0], vj[0]=dp.addCDextra(seqs[0],vj[0],cddict)
        seqs[1], vj[1]=dp.addCDextra(seqs[1],vj[1],cddict)
        #filter
        seqs[0]=dp.filtr(seqs[0], length)
        seqs[1]=dp.filtr(seqs[1], length)
        
        l4=len(seqs[0])
        l8=len(seqs[1])
        
        seqs[0]=createCDR(seqs[0],length=(length-11))
        seqs[1]=createCDR(seqs[1],length=(length-11))
        # each seqs[0] has three dimensions now, each containing the cdrs
        
        c1=dp.char2ptuple(seqs[0][0]+seqs[1][0], n=tupsize)
        c2=dp.char2ptuple(seqs[0][1]+seqs[1][1], n=tupsize)
        c3=dp.char2ptuple(seqs[0][2]+seqs[1][2], n=tupsize)
        
        print("Size CDR3: {}".format(c1.shape[1]))
        print("Size CDR2: {}".format(c2.shape[1]))
        print("Size CDR1: {}".format(c3.shape[1]))
        
        
        X_init=hstack([c1,c2,c3])
        y4=np.zeros((l4))
        y8=np.ones((l8))
        y=np.concatenate((y4,y8))
        X_init=X_init.tocsr()
        # shuffle data
        X, y = sk.utils.shuffle(X_init,y)
    
    else:
        # use function to create data
        X, y = dp.dataCreator(seqs[0],seqs[1])
        
        X=dp.char2ptuple(X, n=tupsize)
        X=X.tocsr()
        
        # shuffle data
        X, y = sk.utils.shuffle(X,y)
    
X=X[:20000]
y=y[:20000]     

X=X.toarray()

 
#==============================================================================
# Model Spec. 
#==============================================================================
    

aeControllerParams={'batch_size':256,
                     'epochs':1000,
                     'learningRate':0.1}
                     
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




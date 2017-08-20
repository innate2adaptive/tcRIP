# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 16:54:22 2017

@author: lewismoffat

This script implements the feature reduction technique from Cinelli et al. 2017
That uses a 1-dimensional Gaussian Naive Bayes to calculate the value of
features.

"""

#==============================================================================
# IMPORTS
#==============================================================================

import dataProcessing as dp
import sklearn as sk
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB

from scipy.sparse import hstack

from xgboost import XGBClassifier

from keras.models import Model
from keras.layers import Input, Dense
    
from sklearn.neighbors import KNeighborsClassifier as KNN

#==============================================================================
# ORGANIZE THE DATA
#==============================================================================

# load in sequences and vj info for all patients with naive and beta chains
seqs, vj = dp.loadAllPatients()

# filter out joint sequences
seqs[0], seqs[1], vj[0], vj[1], joint = dp.removeDup(seqs[0], seqs[1], vj[0], vj[1])

print("Number of Shared Seqs: {}".format(len(joint)))
print("Shared Percent: %.2f%%" % (len(joint)/(len(seqs[0])+len(seqs[1])) * 100.0))


#==============================================================================
# Parameters
#==============================================================================

squish = False
loadCDR12 = False
length = 14
tuplen = 2

print("p = {}".format(tuplen))


#==============================================================================
# Feat. Eng.
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
    
    c1=dp.char2ptuple(seqs[0][0]+seqs[1][0], n=tuplen)
    c2=dp.char2ptuple(seqs[0][1]+seqs[1][1], n=tuplen)
    c3=dp.char2ptuple(seqs[0][2]+seqs[1][2], n=tuplen)
    
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
    
    X=dp.char2ptuple(X, n=tuplen)
    
    # shuffle data
    X, y = sk.utils.shuffle(X,y)
    X=X.toarray()
if squish:
    X=X[:50000]
    y=y[:50000]

# print class balances
dp.printClassBalance(y)

# 20% Validation set
xTrain, xVal, yTrain, yVal= train_test_split(X, y, test_size=0.20) 



#==============================================================================
# Run the feature reduction
#==============================================================================
# we can actually use the Gaussian Naive Bayes implementation from Sklearn as 
# a shortcut for calculating the class priors, means of each feature per class
# and variance. This lets us vectorize some of the calcs down the line

NB = GaussianNB()
# train it
NB.fit(xTrain,yTrain)
# get the values
means =NB.theta_
var   =NB.sigma_
priors=NB.class_prior_

def DBayes(x, mu, sig, prior):
    """
    This takes a set of training data and the corresponding class/feature info
    and calculate for each position the feature importance
    1=CD4
    2=CD8
    """
    termA = np.log(prior[0]/[1])
    







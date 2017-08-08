# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 17:49:59 2017

@author: lewismoffat
"""

#==============================================================================
# Module Imports 
#==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import dataProcessing as dp
import pdb
import sklearn as sk

from collections import defaultdict
from collections import Counter

from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

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


seqs, vj = dp.loadAll(delim) # these gets all the sequences and vj values


#==============================================================================
# Basic Analytics
#==============================================================================

# filter out joint sequences
seqs[0], seqs[1], vj[0], vj[1], joint = dp.removeDup(seqs[0], seqs[1], vj[0], vj[1])

# get only length 14 
seqs[0]=dp.filtr(seqs[0], 14)
seqs[1]=dp.filtr(seqs[1], 14)


#==============================================================================
# Feature Engineering
#==============================================================================

seqs[0]=dp.seq2fatch(seqs[0])
seqs[1]=dp.seq2fatch(seqs[1])

#==============================================================================
# Preparing for Classification
#==============================================================================


index80=int(len(seqs[0])*0.8) # li had 80% CD4, works as classes are both 1
index20=int(len(seqs[1])*0.2) # li had 20% CD8

seqs[0]=seqs[0][:index80]
seqs[1]=seqs[1][:index20]

# use function to create data
X, y = dp.dataCreator(seqs[0],seqs[1])

# shuffle data
X, y = sk.utils.shuffle(X,y)

# print class balances
dp.printClassBalance(y)

#X=X[:1000]
#y=y[:1000]

xTrain, xVal, yTrain, yVal = train_test_split(X, y, test_size=0.20) 


#==============================================================================
# Classify using SVM
#==============================================================================

clf = SVC(C=1, gamma=1e-1, kernel='rbf')

clf.fit(xTrain, yTrain) 

# Prints the validation accuracy
y_true, y_pred = yVal, clf.predict(xVal)
accuracy = accuracy_score(y_true, y_pred)
print("Validation Accuracy: %.2f%%" % (accuracy * 100.0))










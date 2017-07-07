# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 15:55:32 2017

@author: lewismoffat
"""

########################################################
# Module Imports 
########################################################

import numpy as np
import dataProcessing as dp
import pdb
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import AdaBoostClassifier
from matplotlib import pylab
import glob
#from Bio.Blast.Applications import NcbipsiblastCommandline
from collections import defaultdict
import pureModel as n2

########################################################
# Parameters
########################################################


# File names for different data; A - alpha chain, B - beta chain            
cd4A_file = 'patient1/vDCRe_alpha_EG10_CD4_naive_alpha.txt'
cd8A_file = 'patient1/vDCRe_alpha_EG10_CD8_naive_alpha.txt'
cd4B_file = 'patient1/vDCRe_beta_EG10_CD4_naive_beta.txt'
cd8B_file = 'patient1/vDCRe_beta_EG10_CD8_naive_beta.txt'
data = 'data/'
extra = 'extra/'


# This runs adaboost with atcheley and with v region attached
secondary=True
afterV=False
nn=True
svm=True

########################################################
# Data Retrieval 
########################################################
"""
The data for the first patient is stored in the 'data/patient1' file
where each sequence is encoded as a string and comma separated with its count
at current extraction is unique and count is ignored
"""

# Files to be read
files = [cd4B_file, cd8B_file]

# sequence list to be filled. Make sure file order is the same as a below
cd4=[]
cd8=[]
seqs=[cd4,cd8]   # this contains the sequences 
cd4vj=[]
cd8vj=[]
vj=[cd4vj,cd8vj] # this contains the v and j index genes

for index, file in enumerate(files):
        file=data+extra+file
        with open(file,'r') as infile:
            # goes through each of the files specified in read mode and pulls out 
            # each line adds each sequence from the line 
            for line in infile:
                threeVals=line.split(",")
                threeVals[2]=threeVals[2].replace("\n","")
                seqs[index].append(threeVals)


########################################################
# CDR1 Length Analytics
########################################################
                
# length histogram
lenArr=[]
# create new vector of lengths 
for seq in cd4+cd8:
    lenArr.append(len(seq[1]))
    
# convert list to numpy array
lenArr=np.asarray(lenArr)

# bins
binNum=max(lenArr)-min(lenArr)
bins=np.arange(min(lenArr),max(lenArr))

# setup graph
plt.hist(lenArr, bins=binNum)
plt.xticks(bins)
plt.title("CDR1 Sequence Length Histogram")
plt.xlabel("Sequence Length (AA)")
plt.ylabel("Frequency")
plt.show()
       
########################################################
# CDR1 Classification
########################################################         

# separate out cdr1s
CDR1_4=[]
CDR1_8=[]

or12=0# this specs either cdr1 or cdr2
limit=5

for row in cd4:
    if len(row[or12])==limit and len(row[1])==6: CDR1_4.append(row[or12])
for row in cd8:
    if len(row[or12])==limit and len(row[1])==6: CDR1_8.append(row[or12])


CDR1_4=dp.seq2fatch(CDR1_4)
CDR1_8=dp.seq2fatch(CDR1_8)


# use function to create data
X_1, y_1 = dp.dataCreator(CDR1_4,CDR1_8)

# shuffle data
X, y = sk.utils.shuffle(X_1,y_1)

# print class balances
dp.printClassBalance(y)

# 25% Validation set
xTrain, xVal, yTrain, yVal= train_test_split(X, y, test_size=0.25) 

# Set up an adaboost Classifer
clf = AdaBoostClassifier(n_estimators=200)

# Fit the booster
clf.fit(xTrain, yTrain)

# Prints the validation accuracy
y_true, y_pred = yVal, clf.predict(xVal)
print("{} Validaton Accuracy".format(accuracy_score(y_true, y_pred)))



########################################################         
# CDR2 Classification
########################################################         

# separate out cdr1s
CDR1_4_2=[]
CDR1_8_2=[]

or12=1# this specs either cdr1 or cdr2
limit=6

for row in cd4:
    if len(row[or12])==limit and len(row[0])==5: CDR1_4_2.append(row[or12])
for row in cd8:
    if len(row[or12])==limit and len(row[0])==5: CDR1_8_2.append(row[or12])


CDR1_4_2=dp.seq2fatch(CDR1_4_2)
CDR1_8_2=dp.seq2fatch(CDR1_8_2)


# use function to create data
X_2, y_2 = dp.dataCreator(CDR1_4_2,CDR1_8_2)

# shuffle data
X, y = sk.utils.shuffle(X_2,y_2)

# print class balances
dp.printClassBalance(y)

# 25% Validation set
xTrain, xVal, yTrain, yVal= train_test_split(X, y, test_size=0.25) 

# Set up an adaboost Classifer
clf1 = AdaBoostClassifier(n_estimators=200)

# Fit the booster
clf1.fit(xTrain, yTrain)

# Prints the validation accuracy
y_true, y_pred = yVal, clf1.predict(xVal)
print("{} Validaton Accuracy".format(accuracy_score(y_true, y_pred)))

########################################################         
# Ensemble using probs as features Classification
########################################################    

# shuffle data
X_1,y_1,X_2,y_2 = sk.utils.shuffle(X_1,y_1,X_2,y_2)

# Prints the validation accuracy
X_1= clf.predict_log_proba(X_1)
X_2 = clf1.predict_log_proba(X_2)

X=np.concatenate((X_1,X_2),1)

# 25% Validation set
xTrain, xVal, yTrain, yVal= train_test_split(X,y_1, test_size=0.25) 

pdb.set_trace()

clf_s = SVC(C=10, kernel='rbf', gamma=1e-3, class_weight='balanced',decision_function_shape='ovr')

# Fit the svm
clf_s.fit(xTrain, yTrain)
# Prints the cross validation report for the different parameters

y_true, y_pred = yVal, clf_s.predict(xVal)
print("{} Validaton Accuracy".format(accuracy_score(y_true, y_pred)))





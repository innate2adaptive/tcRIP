# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 14:40:52 2017

@author: lewismoffat
"""

########################################################
# Module Imports 
########################################################

import numpy as np
import dataProcessing as dp
import pdb
import sklearn as sk
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
#from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import AdaBoostClassifier
#from matplotlib import pylab
#import glob
#from Bio.Blast.Applications import NcbipsiblastCommandline
from collections import defaultdict
import pureModel as n2
from xgboost import XGBClassifier
import glob
from sklearn.tree import DecisionTreeClassifier

from imblearn.ensemble import BalanceCascade
from imblearn import pipeline as pl
from imblearn.metrics import classification_report_imbalanced
from imblearn.metrics import (geometric_mean_score,
                              make_index_balanced_accuracy)
from imblearn import over_sampling as os
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif 
from sklearn.naive_bayes import GaussianNB  
from sklearn.ensemble import VotingClassifier

########################################################
# Parameters
########################################################


# File names for different data; A - alpha chain, B - beta chain            
cd4A_file = 'patient1/vDCRe_alpha_EG10_CD4_naive_alpha.dcrcdr3'
cd8A_file = 'patient1/vDCRe_alpha_EG10_CD8_naive_alpha.dcrcdr3'
cd4B_file = 'patient1/vDCRe_beta_EG10_CD4_naive_beta.dcrcdr3'
cd8B_file = 'patient1/vDCRe_beta_EG10_CD8_naive_beta.dcrcdr3'
data = 'data/'
extra = 'VandJ/'

# This runs adaboost with atcheley and with v region attached
original= False 
secondary=True # This is the best
afterV=False
nn=False
svm=False
EG10=True# this means only data from patient EG10
feature_select=False
    
########################################################
# Data Retrieval 
########################################################
"""
The data for the first patient is stored in the 'data/patient1' file
where each sequence is encoded as a string and comma separated with its count
at current extraction is unique and count is ignored
"""
if EG10:
    # Files to be read
    files = [cd4B_file, cd8B_file]
    
    # sequence list to be filled. Make sure file order is the same as a below
    cd4=[]
    cd8=[]
    seqs=[cd4,cd8]   # this contains the sequences 
    cd4vj=[]
    cd8vj=[]
    vj=[cd4vj,cd8vj] # this contains the v and j index genes
    ex4=[]
    ex8=[]
    extraVals=[ex4, ex8]
    
    
    for index, file in enumerate(files):
        file=data+extra+file
        with open(file,'r') as infile:
            # goes through each of the files specified in read mode and pulls out 
            # each line and formats it so a list gets X copies of the sequence 
            for line in infile:
                twoVals=line.split(":")
                twoVals[1]=twoVals[1].replace("\n","")
                twoVals[1]=twoVals[1].split(",")
                twoVals[0]=twoVals[0].split(",")
                seqs[index].append(twoVals[1][0])
                vj[index].append(twoVals[0][:1])
                extraVals[index].append(twoVals[0][:4])

else:
    files=glob.glob("F:/seqs/*.txt")
    cd4, cd8 = dp.dataReader(files, ["naive","beta"])
    
    # focus on just sequence and v region
    cd4,cd4vj = dp.dataSpliter(cd4)
    cd8,cd8vj = dp.dataSpliter(cd8)
    
    # sequence list to be filled. Make sure file order is the same as a below
    seqs=[cd4,cd8]   # this contains the sequences 
    vj=[cd4vj,cd8vj] 
    

########################################################
# Data Retrieval 
########################################################
print("Training Algorithm")
    
    
# filter out joint sequences
seqs[0], seqs[1], vj[0], vj[1], joint = dp.removeDup(seqs[0], seqs[1], vj[0], vj[1])

print("Number of Shared Seqs: {}".format(len(joint)))
print("Shared Percent: %.2f%%" % (len(joint)/(len(seqs[0])+len(seqs[1])) * 100.0))

# add extrac cds
cddict=dp.extraCDs()
seqs[0], vj[0]=dp.addCDextra(seqs[0],vj[0],cddict)
seqs[1], vj[1]=dp.addCDextra(seqs[1],vj[1],cddict)


# replace the seqeunces with their atchely vectors and add the v region as the
# final region
for idx, seqz in enumerate(seqs):
    seqs[idx]=dp.seq2fatch(seqz) 
    for idx2, seq in enumerate(seqs[idx]):
        new=np.zeros((1))
        new[0]=float(vj[idx][idx2][0])
        seqs[idx][idx2]=np.concatenate((seq,new))

# get the dictionaries that map v values and j values to (sub)families 
jDict, vDict= dp.getVJnames()
# go through each row and concatenate those two extra values on

for idx2, seq in enumerate(seqs[0]):
    new=vDict[seq[-1]] # last value is the v index
    seqs[0][idx2]=np.concatenate((seq,new))

for idx2, seq in enumerate(seqs[1]):
    new=vDict[seq[-1]] # last value is the v index
    seqs[1][idx2]=np.concatenate((seq,new))

pdb.set_trace()
        
        
        
# filter to a set length
length=14+5+6 # 14 is the most abundant
seqs[0]=np.array(dp.filtr(seqs[0], length*5+1+2))
seqs[1]=np.array(dp.filtr(seqs[1], length*5+1+2))


# calculate sequence lengths for tf models
sq4=[len(x) for x in seqs[0]]
sq8=[len(x) for x in seqs[1]]
sqlen=np.concatenate((sq4,sq8))

# use function to create data
X, y = dp.dataCreator(seqs[0],seqs[1])

# shuffle data
X, y = sk.utils.shuffle(X,y)


if feature_select:    
    sel=SelectKBest(mutual_info_classif, k=110)
    X = sel.fit_transform(X, y)
    


# print class balances
dp.printClassBalance(y)

# 25% Validation set
xTrain, xVal, yTrain, yVal, sqTrain, sqVal= train_test_split(X, y, sqlen, test_size=0.25) 


########################################################
# Classification including v as a feature using Adaboost
########################################################
print("======================================")
print("Running Classification using AdaBoost")
# Set up an adaboost Classifer
clf = AdaBoostClassifier(n_estimators=100)

# Fit the booster
clf.fit(xTrain, yTrain)

# Prints the validation accuracy
y_true, y_pred = yVal, clf.predict(xVal)
accuracy = accuracy_score(y_true, y_pred)
print("Validation Accuracy: %.2f%%" % (accuracy * 100.0))

########################################################
# Classification including v as a feature using XGBoost
########################################################
print("======================================")
print("Running Classification using XGBoost")

model = XGBClassifier()
model.fit(xTrain, yTrain)
#print(model)
y_pred = model.predict(xVal)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(yVal, predictions)
print("Validation Accuracy: %.2f%%" % (accuracy * 100.0))




########################################################
# Classification using All Info: SVM
########################################################

print("======================================")
print("Classification using Decision Tree Classifier")


# naive bayes so we got three classifiers
clf3 = DecisionTreeClassifier()

# Fit the classifier
clf3.fit(xTrain, yTrain)

# Prints the validation accuracy
y_true, y_pred = yVal, clf3.predict(xVal)
accuracy = accuracy_score(y_true, y_pred)
print("Validation Accuracy: %.2f%%" % (accuracy * 100.0))

########################################################
# Classification using All Info: Voting Ensemble
########################################################
print("======================================")
print("Classification using Ensemble: KNN+XGBoost+AdaBoost")


clf1 = AdaBoostClassifier()
clf2 = XGBClassifier()
clf3 = DecisionTreeClassifier()
eclf1 = VotingClassifier(estimators=[('ab', clf1), ('xg', clf2), ('knn', clf3)], voting='hard')

eclf1.fit(xTrain, yTrain)

# Prints the validation accuracy
y_true, y_pred = yVal, eclf1.predict(xVal)
accuracy = accuracy_score(y_true, y_pred)
print("Validation Accuracy: %.2f%%" % (accuracy * 100.0))









########################################################
# Classification using ensemble of correctly balanced classes
########################################################
#print("======================================")
#print("Running Balanced Classification using Adaboost")
#
## balance cascade produces a bunch of classes for the estimators that are 
## even
#pipeline = pl.make_pipeline(os.SMOTE(),
#                        AdaBoostClassifier(n_estimators=200))
## 25% Validation set
#xTrain, xVal, yTrain, yVal, sqTrain, sqVal= train_test_split(X, y, sqlen, test_size=0.25) 
#
## Train the classifier with balancing
#pipeline.fit(xTrain, yTrain)
#
## Test the classifier and get the prediction
#y_pred_bal = pipeline.predict(xVal)
#
## Show the classification report
#print(classification_report_imbalanced(yVal, y_pred_bal))
#    
#print('The geometric mean is {}'.format(geometric_mean_score(
#yVal,
#y_pred_bal)))

########################################################
# Classification using V(D)J Info which is included in extraVals list
########################################################
print("======================================")
print("Classification using V and J families")


jDict, vDict= dp.getVJnames()

extraVals[0]=np.array(extraVals[0]).astype(int)
extraVals[1]=np.array(extraVals[1]).astype(int)


def addSubClasses(seqs,jDict,vDict):
    # just get the extra classes and sub classes and wack it on
    classes=np.zeros_like(seqs)
    for idx, row in enumerate(classes):
        # first two values are the v values 
        vIndex=seqs[idx][0]
        jIndex=seqs[idx][1]
        jValues=jDict[jIndex]
        vValues=vDict[vIndex]
        classes[idx]=[vValues[0],vValues[1],jValues[0],jValues[1]]
    return np.concatenate((seqs,classes),1)

extraVals[0]=addSubClasses(extraVals[0],jDict,vDict)
extraVals[1]=addSubClasses(extraVals[1],jDict,vDict)


########################################################
# Classification using V(D)J Info: AdaBoost
########################################################

# make data
X, y = dp.dataCreator(extraVals[0],extraVals[1])

# shuffle data
X, y = sk.utils.shuffle(X,y)

# 25% Validation set
xTrain, xVal, yTrain, yVal= train_test_split(X, y, test_size=0.25)

# Set up an adaboost Classifer
clf1 = AdaBoostClassifier(n_estimators=100)

# Fit the booster
clf1.fit(xTrain, yTrain)

# Prints the validation accuracy
y_true, y_pred = yVal, clf1.predict(xVal)
accuracy = accuracy_score(y_true, y_pred)
print("Validation Accuracy: %.2f%%" % (accuracy * 100.0))

########################################################
# Classification using V(D)J Info: XGBoost
########################################################

print("======================================")
print("Classification using XGBoost Classifier")


# naive bayes so we got three classifiers
clf2 = XGBClassifier()

# Fit the classifier
clf2.fit(xTrain, yTrain)

# Prints the validation accuracy
y_true, y_pred = yVal, clf2.predict(xVal)
accuracy = accuracy_score(y_true, y_pred)
print("Validation Accuracy: %.2f%%" % (accuracy * 100.0))









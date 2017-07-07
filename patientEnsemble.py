# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 16:25:46 2017

@author: lewismoffat
"""
########################################################
# Module Imports 
########################################################

import numpy as np
import dataProcessing as dp
#import rnnModel as r2n
import pureModel as n2
import autoencoderModel as ae
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



########################################################
# Get all Files from USB
########################################################

files=glob.glob("F:/seqs/*.cdr3")

# sequence list to be filled. Make sure file order is the same as a below
cd4_EG10=[]
cd8_EG10=[]

cd4_SK11=[]
cd8_SK11=[]

cd4_KS07=[]
cd8_KS07=[]

allSeqs=[cd4_EG10,cd8_EG10,cd4_SK11,cd8_SK11,cd4_KS07,cd8_KS07]

def dataExtractor(file):
    # this extracts unique CDR3s from .cdr3 files that arnt gzipped
    seqz=[]
    with open(file,'r') as infile:
        # goes through each of the files specified in read mode and pulls out 
        # each line and formats it so a list gets X copies of the sequence 
        for line in infile:
            twoVals=line.split(", ")
            twoVals[1]=twoVals[1].replace("\n","")
            seqz.append(twoVals[0])
    return seqz
# Three patients KS07, SK11, EG10

for fil in files:
    if "beta" in fil and "naive" in fil and "KS07" in fil:
        print(fil)
        seqz=dataExtractor(fil)
        if "CD4" in fil:
            for seq in seqz: 
                if len(seq)==14: 
                    cd4_KS07.append(seq)
        else:
            for seq in seqz: 
                if len(seq)==14: 
                    cd8_KS07.append(seq)
            
    if "beta" in fil and "naive" in fil and "EG10" in fil:
        print(fil)
        seqz=dataExtractor(fil)
        if "CD4" in fil:
            for seq in seqz: 
                if len(seq)==14:
                    cd4_EG10.append(seq)
        else:
            for seq in seqz: 
                if len(seq)==14:
                    cd8_EG10.append(seq)
    if "beta" in fil and "naive" in fil and "SK11" in fil:
        print(fil)
        seqz=dataExtractor(fil)
        if "CD4" in fil:
            for seq in seqz:
                if len(seq)==14: 
                    cd4_SK11.append(seq)
        else:
            for seq in seqz:
                if len(seq)==14: 
                    cd8_SK11.append(seq)
        
########################################################
# Feature Engineering
########################################################
for idx, seqs in enumerate(allSeqs):
    allSeqs[idx]=dp.seq2fatch(seqs)            

y4_EG10=np.zeros((len(cd4_EG10)))
y8_EG10=np.zeros((len(cd8_EG10)))
y4_SK11=np.zeros((len(cd4_SK11)))
y8_SK11=np.zeros((len(cd8_SK11)))
y4_KS07=np.zeros((len(cd4_KS07)))
y8_KS07=np.zeros((len(cd8_KS07)))
y4_EG10[:]=0
y8_EG10[:]=1
y4_SK11[:]=0
y8_SK11[:]=1
y4_KS07[:]=0
y8_KS07[:]=1



# Print class copositions 
print("EG10 {}:{}".format(len(y4_EG10)/(len(y4_EG10)+len(y8_EG10)), len(y8_EG10)/(len(y4_EG10)+len(y8_EG10))))
print("SK11 {}:{}".format(len(y4_SK11)/(len(y4_SK11)+len(y8_SK11)), len(y8_SK11)/(len(y4_SK11)+len(y8_SK11))))
print("KS07 {}:{}".format(len(y4_KS07)/(len(y4_KS07)+len(y8_KS07)), len(y8_KS07)/(len(y4_KS07)+len(y8_KS07))))



# combine classes
Y_EG10 = np.concatenate((y4_EG10,y8_EG10),0)
X_EG10 = np.concatenate((cd4_EG10,cd8_EG10),0)

Y_SK11 = np.concatenate((y4_SK11,y8_SK11),0)
X_SK11 = np.concatenate((cd4_SK11,cd8_SK11),0)

Y_KS07 = np.concatenate((y4_KS07,y8_KS07),0)
X_KS07 = np.concatenate((cd4_KS07,cd8_KS07),0)
    
xTrain_EG10, xVal_EG10, yTrain_EG10, yVal_EG10 = train_test_split(X_EG10, Y_EG10, test_size=0.20) 
# memory clean up
X_EG10=None
Y_EG10=None 

xTrain_SK11, xVal_SK11, yTrain_SK11, yVal_SK11 = train_test_split(X_SK11, Y_SK11, test_size=0.20) 
# memory clean up
X_SK11=None
Y_SK11=None 

xTrain_KS07, xVal_KS07, yTrain_KS07, yVal_KS07 = train_test_split(X_KS07, Y_KS07, test_size=0.20) 
# memory clean up
X_KS07=None
Y_KS07=None 
    
########################################################
# EG10 SVM
########################################################

tuned_parameters = [{'kernel': ['rbf'], 'C': [10], 'gamma':[1e-3]}]
# runs grid search using the above parameter and doing 5-fold cross validation
#clf_EG10 = GridSearchCV(SVC(C=1, class_weight='balanced',decision_function_shape='ovr'), tuned_parameters, cv=2, verbose=1)
clf_EG10 = AdaBoostClassifier(n_estimators=200)

# Fit the svm
clf_EG10.fit(xTrain_EG10, yTrain_EG10)
# Prints the cross validation report for the different parameters

y_true, y_pred = yVal_EG10, clf_EG10.predict(xVal_EG10)
print("{} Validaton Accuracy EG10".format(accuracy_score(y_true, y_pred)))


########################################################
# KS07 SVM
########################################################

# runs grid search using the above parameter and doing 5-fold cross validation
#clf_KS07 = SVC(C=10,gamma=1e-3,kernel='rbf', class_weight='balanced',decision_function_shape='ovr', verbose=2) #GridSearchCV(SVC(C=1, class_weight='balanced',decision_function_shape='ovr'), tuned_parameters, cv=2, verbose=1)

clf_KS07 = AdaBoostClassifier(n_estimators=200)

# Fit the svm
clf_KS07.fit(xTrain_KS07, yTrain_KS07)
# Prints the cross validation report for the different parameters
y_true, y_pred = yVal_KS07, clf_KS07.predict(xVal_KS07)
print("{} Validaton Accuracy KS07".format(accuracy_score(y_true, y_pred)))



########################################################
# SK11 SVM
########################################################

tuned_parameters = [{'kernel': ['rbf'], 'C': [10], 'gamma':[1e-3]}]
# runs grid search using the above parameter and doing 5-fold cross validation
#clf_SK11 = GridSearchCV(SVC(C=1, class_weight='balanced',decision_function_shape='ovr'), tuned_parameters, cv=2, verbose=1)
clf_SK11 = AdaBoostClassifier(n_estimators=200)

# Fit the svm
clf_SK11.fit(xTrain_SK11, yTrain_SK11)
# Prints the cross validation report for the different parameters

y_true, y_pred = yVal_SK11, clf_SK11.predict(xVal_SK11)
print("{} Validaton Accuracy SK11".format(accuracy_score(y_true, y_pred)))



########################################################
# Final Test 
########################################################

y_pred_SK11 = clf_SK11.predict_proba(xVal_SK11)
y_pred_EG10 = clf_EG10.predict_proba(xVal_SK11)
y_pred_KS07 = clf_KS07.predict_proba(xVal_SK11)


y_pred = np.multiply(y_pred_SK11,y_pred_EG10,y_pred_KS07)
y_pred = np.argmax(y_pred, axis=1)
print("{} SK11 Combo'd Validaton Accuracy".format(accuracy_score(yVal_SK11, y_pred)))



y_pred_SK11 = clf_SK11.predict_proba(xVal_EG10)
y_pred_EG10 = clf_EG10.predict_proba(xVal_EG10)
y_pred_KS07 = clf_KS07.predict_proba(xVal_EG10)


y_pred = np.multiply(y_pred_SK11,y_pred_EG10,y_pred_KS07)
y_pred = np.argmax(y_pred, axis=1)
print("{} EG10 Combo'd Validaton Accuracy".format(accuracy_score(yVal_EG10, y_pred)))



y_pred_SK11 = clf_SK11.predict_proba(xVal_KS07)
y_pred_EG10 = clf_EG10.predict_proba(xVal_KS07)
y_pred_KS07 = clf_KS07.predict_proba(xVal_KS07)


y_pred = np.multiply(y_pred_SK11,y_pred_EG10,y_pred_KS07)
y_pred = np.argmax(y_pred, axis=1)
print("{} KS07 Combo'd Validaton Accuracy".format(accuracy_score(yVal_KS07, y_pred)))












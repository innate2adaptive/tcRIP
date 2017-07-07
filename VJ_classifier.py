# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 12:09:09 2017

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
cd4A_file = 'patient1/vDCRe_alpha_EG10_CD4_naive_alpha.dcrcdr3'
cd8A_file = 'patient1/vDCRe_alpha_EG10_CD8_naive_alpha.dcrcdr3'
cd4B_file = 'patient1/vDCRe_beta_EG10_CD4_naive_beta.dcrcdr3'
cd8B_file = 'patient1/vDCRe_beta_EG10_CD8_naive_beta.dcrcdr3'
data = 'data/'
extra = 'VandJ/'

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
        # each line and formats it so a list gets X copies of the sequence 
        for line in infile:
            twoVals=line.split(":")
            twoVals[1]=twoVals[1].replace("\n","")
            twoVals[1]=twoVals[1].split(",")
            twoVals[0]=twoVals[0].split(",")
            seqs[index].append(twoVals[1][0])
            vj[index].append(twoVals[0][:1])
            
# data is in the format [v_index, j_index, deletionsV, deletionsJ, extra:CDR3, count]

########################################################
# Classification based on just v_index and j_index
########################################################
"""
This takes the only the v and classifys upon it
"""

# use function to create data
X, y = dp.dataCreator(vj[0],vj[1])

# shuffle data
X, y = sk.utils.shuffle(X,y)

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
# Classification including v as a feature with traditional 
########################################################

if secondary:
    print("Training Second Algorithm")
    
    # replace the seqeunces with their atchely vectors and add the v region as the
    # final region
    for idx, seqz in enumerate(seqs):
        seqs[idx]=dp.seq2fatch(seqz) 
        for idx2, seq in enumerate(seqs[idx]):
            new=np.zeros((1))
            new[0]=float(vj[idx][idx2][0])
            seqs[idx][idx2]=np.concatenate((seq,new))
        
     
    # filter to a set length
    length=14 # 14 is the most abundant
    seqs[0]=np.array(dp.filtr(seqs[0], length*5+1))
    seqs[1]=np.array(dp.filtr(seqs[1], length*5+1))
    
    # calculate sequence lengths for tf models
    sq4=[len(x) for x in seqs[0]]
    sq8=[len(x) for x in seqs[1]]
    sqlen=np.concatenate((sq4,sq8))
    
    # use function to create data
    X, y = dp.dataCreator(seqs[0],seqs[1])
    
    # shuffle data
    X, y = sk.utils.shuffle(X,y)
    
    # print class balances
    dp.printClassBalance(y)
    
    # 25% Validation set
    xTrain, xVal, yTrain, yVal, sqTrain, sqVal= train_test_split(X, y, sqlen, test_size=0.25) 
    
    
    
    # Set up an adaboost Classifer
    clf = AdaBoostClassifier(n_estimators=200)
    
    # Fit the booster
    clf.fit(xTrain, yTrain)
    
    # Prints the validation accuracy
    y_true, y_pred = yVal, clf.predict(xVal)
    print("{} Validaton Accuracy".format(accuracy_score(y_true, y_pred)))
    
    
    if svm:
        # grid search for best parameters for both linear and rbf kernels
        tuned_parameters = [{'kernel': ['rbf'], 'C': [0.1,1,10,100], 'gamma':[1e-3,1e-2,1e-1]}]
        # runs grid search using the above parameter and doing 5-fold cross validation
        clf = GridSearchCV(SVC(C=1, class_weight='balanced',decision_function_shape='ovr'), tuned_parameters, cv=2, verbose=1)
        print(clf)
        # Fit the svm
        clf.fit(xTrain, yTrain)
        # Prints the cross validation report for the different parameters
        print("Best parameters set found on validation set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()        
        y_true, y_pred = yVal, clf.predict(xVal)
        print(classification_report(y_true, y_pred))
        print()    
        
        
    
    if nn:
        ModelParams={'learningRate':0.001,
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
        
        ControllerParams={'batch_size':256,
                         'epochs':100}
                         
        # Spool up warp drives! This gets the normal nn controller class going
        ModelParams['maxLen']=xTrain.shape[1]
        nnMain=n2.Controller(ControllerParams, ModelParams) 
        print("Training NN")
        nnMain.train(xTrain, yTrain, sqTrain, xVal, yVal, sqVal)

########################################################
# Classification including v as a post feature 
########################################################

if afterV:
    # replace the seqeunces with their atchely vectors and add the v region as the
    # final region
    for idx, seqz in enumerate(seqs):
        seqs[idx]=dp.seq2fatch(seqz) 
        for idx2, seq in enumerate(seqs[idx]):
            seqs[idx][idx2]=np.concatenate((seq,vj[idx][idx2]))
        
     
    # filter to a set length
    length=14 # 14 is the most abundant
    seqs[0]=np.array(dp.filtr(seqs[0], length*5+1))
    seqs[1]=np.array(dp.filtr(seqs[1], length*5+1))
            
    
    # use function to create data
    X, y = dp.dataCreator(seqs[0],seqs[1])
    
    # shuffle data
    X, y = sk.utils.shuffle(X,y)
    
    # rip out the v
    
    v=X[:,length*5]
    X=X[:,:length*5]
    
    
    
    # 25% Validation set
    xTrain, xVal, yTrain, yVal, vTrain, vVal= train_test_split(X, y, v, test_size=0.25) 
    
    # print class balances
    dp.printClassBalance(yTrain)
    
    # Set up an adaboost Classifer
    clf = AdaBoostClassifier(n_estimators=200)
    
    # Fit the booster
    clf.fit(xTrain, yTrain)
    
    # Prints the validation accuracy
    y_true, y_pred = yVal, clf.predict_proba(xVal)
    #print("{} Validaton Accuracy".format(accuracy_score(y_true, y_pred)))
    
    ones={}
    zeros={}
    vs = defaultdict(list)
    
    for idx, val in enumerate(y):
        ones[v[idx]]=0
        zeros[v[idx]]=0
    
    for idx, val in enumerate(y):
        if val==1:
            ones[v[idx]]+=1
        else:
            zeros[v[idx]]+=1
    
    for key in zeros.keys():
        denom=ones[key]+zeros[key]
        new=[ones[key]/denom, zeros[key]/denom]
        vs[key]=new
    
    for idx, row in enumerate(y_pred):
        probs=np.multiply(row, vs[vVal[idx]])
        y_pred[idx]=probs
    
    new = np.argmax(y_pred,1)
    for idx, val in enumerate(new):
        if val==1:
            new[idx]=0
        else:
            new[idx]=1
    
    
    print("{} Validaton Accuracy".format(accuracy_score(y_true, new)))
    

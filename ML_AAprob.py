# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 21:02:49 2017

@author: lewismoffat
"""

#==============================================================================
# IMPORTS
#==============================================================================

import dataProcessing as dp
import sklearn as sk
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN

from scipy.sparse import hstack

from xgboost import XGBClassifier

from keras.models import Model
from keras.layers import Input, Dense
from keras import optimizers
  

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

squish = True
loadCDR12 = False
length = 14



#==============================================================================
# Helper functions
#==============================================================================

# This is used quite often so laying it here
intDictzero={ 'A': 0 ,'C': 0 ,'D': 0 ,'E': 0 ,'F': 0 ,'G': 0 ,'H': 0 ,'I': 0 ,'K': 0 ,'L': 0 ,'M': 0 ,
              'N': 0 ,'P': 0 ,'Q': 0 ,'R': 0 ,'S': 0 ,'T': 0 ,'V': 0 ,'W': 0 ,'Y': 0 ,}

def positionProb(seqs, length, intDictzero):
    # this function takes a list of sequences and for each position it calculates
    # a distribution
    
    # we only want to look at each sequence once so we want to have the e.g. 14
    # positional dictionaries already made
    
    # this stores the dictionaries
    listDict=[]
    for i in range(length):
        # for every position add a dictionary copy of intDictzero
        listDict.append(intDictzero.copy())
    # iterate through every sequence
    for seq in seqs:
        # iterate through every amino acid 
        for idx, aa in enumerate(seq):
            # use its position to access the dict and add one for the amino acid present
            listDict[idx][aa]+=1
    
    dictLookup=[]
    # now take the values and create a numpy array while normalizing 
    for idx, dictionary in enumerate(listDict):
        # this normalizes the values
        factor=1.0/sum(dictionary.values())
        for k in dictionary:
          dictionary[k] = dictionary[k]*factor
        
        dictLookup.append(dictionary)
        
        newVals=[]
        for char in sorted(dictionary):# this enforces alphabetical ordering
            newVals.append(dictionary[char])
        listDict[idx]=np.array(newVals)
        # turns dict values into a numpy array and then replaces the original
        #listDict[idx]=np.fromiter(iter(dictionary.values()), dtype=float)
        #pdb.set_trace()
    listDict=np.array(listDict)
        
    return listDict, dictLookup

# funtion that replaces each sequence with its probability position values
def charForProb(seqs,dic):
    newSeqs=[] # will contain all new sequences
    for seq in seqs:
        run=[] # will contain the new prob vals for a sequence
        for idx, char in enumerate(seq):
            run.append(dic[idx][char]) # go through and replace char with prob
        #run=np.log(np.product(run)+1) # can add if you want it to be stable
        newSeqs.append(run)
    return newSeqs

#==============================================================================
# Feature Engineering    
#==============================================================================
    
    
if loadCDR12:
    length=length+5+6
    # add extrac cds
    cddict=dp.extraCDs()
    seqs[0], vj[0]=dp.addCDextra(seqs[0],vj[0],cddict)
    seqs[1], vj[1]=dp.addCDextra(seqs[1],vj[1],cddict)
        
seqs[0]=dp.filtr(seqs[0], length)
seqs[1]=dp.filtr(seqs[1], length)    
    
# its handy to have the look up dictionaries for 
Prob4, dl4=positionProb(seqs[0], length, intDictzero)
Prob8, dl8=positionProb(seqs[1], length, intDictzero)
    
 
seqs[0]=np.concatenate((charForProb(seqs[0],dl4),charForProb(seqs[0],dl8)),1)
seqs[1]=np.concatenate((charForProb(seqs[1],dl4),charForProb(seqs[1],dl8)),1)

# use function to create data
X, y = dp.dataCreator(seqs[0],seqs[1])

   
# shuffle data
X, y = sk.utils.shuffle(X,y)
    
if squish:
    X=X[:10]
    y=y[:10]

# print class balances
dp.printClassBalance(y)

# 20% Validation set
xTrain, xVal, yTrain, yVal= train_test_split(X, y, test_size=0.20) 


##==============================================================================
## Adaboost
##==============================================================================

print("======================================")
print("Running Classification using AdaBoost")
# Set up an adaboost Classifer
clf = AdaBoostClassifier(n_estimators=100)
print(clf)
# Fit the booster
clf.fit(xTrain, yTrain)

# Prints the validation accuracy
y_true, y_pred = yVal, clf.predict(xVal)
accuracy = accuracy_score(y_true, y_pred)
print("Validation Accuracy: %.2f%%" % (accuracy * 100.0))


#==============================================================================
# XGBoost
#==============================================================================
print("======================================")
print("Running Classification using XGBoost")

model = XGBClassifier(reg_lambda=1)
print(model)
model.fit(xTrain, yTrain)

y_pred = model.predict(xVal)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(yVal, predictions)
print("Validation Accuracy: %.2f%%" % (accuracy * 100.0))

#==============================================================================
# SVM
#==============================================================================
print("======================================")
print("Running Classification using SVM")
## grid search for best parameters for both linear and rbf kernels
##tuned_parameters = [{'kernel': ['rbf'], 'C': [0.1,1,10,100], 'gamma':[1e-3,1e-2,1e-1]}]
## runs grid search using the above parameter and doing 5-fold cross validation
##clf = GridSearchCV(SVC(C=1, gamma=0.01, class_weight='balanced',decision_function_shape='ovr'), tuned_parameters, cv=2, verbose=1)
##print(clf)
## Fit the svm
#
#
clf=SVC(C=1, gamma=0.01, class_weight='balanced',decision_function_shape='ovr')
#n_estimators=10
##clf=BaggingClassifier(SVC(kernel='linear', class_weight='balanced'), max_samples=1.0 / n_estimators, n_estimators=n_estimators)
#

clf.fit(xTrain, yTrain)

## Prints the cross validation report for the different parameters
#print("Best parameters set found on validation set:")
##print()
##print(clf.best_params_)
##print()
##print("Grid scores on development set:")
##print()
##means = clf.cv_results_['mean_test_score']
##stds = clf.cv_results_['std_test_score']
##for mean, std, params in zip(means, stds, clf.cv_results_['params']):
##    print("%0.3f (+/-%0.03f) for %r"
##          % (mean, std * 2, params))
##print()    
    
y_true, y_pred = yVal, clf.predict(xVal)

#print(classification_report(y_true, y_pred))
#
#print()    

accuracy = accuracy_score(y_true, y_pred)
print("Validation Accuracy: %.2f%%" % (accuracy * 100.0))

#==============================================================================
# NN
#==============================================================================

def quickExpand(y):
    newY=[]
    for val in y:
        small=[0,0]
        small[int(val)]+=1
        newY.append(small)
    newY=np.array(newY)
    return newY
    
y=quickExpand(y)


# 25% Validation set
xTrain, xVal, yTrain, yVal= train_test_split(X, y, test_size=0.20) 


# defining the keras model
inCDR3 = Input(shape=(xTrain.shape[1],))

#CDR1 pipeline
layer1 = Dense(512, activation='relu')(inCDR3)
layer2 = Dense(128, activation='relu')(layer1)
layer3 = Dense(32, activation='relu')(layer2)
predictions = Dense(2, activation='softmax')(layer3)

# define the model
model = Model(inputs=inCDR3, outputs=[predictions])

sgd = optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x=xTrain, 
          y=yTrain, 
          batch_size=256, 
          epochs=10, 
          verbose=1,
          validation_split=0.2)

# evaluate the model
scores = model.evaluate(xVal, yVal)
print("\nTesting Accuracy: %.2f%%" % ( scores[1]*100))

#==============================================================================
# k-NN
#==============================================================================

print("======================================")
print("Running Classification using k-NN")

model = KNN(n_neighbors=5)

model.fit(xTrain, yTrain)
#print(model)   
y_pred = model.predict(xVal)
predictions = y_pred

accuracy = accuracy_score(yVal, predictions)
print("Validation Accuracy: %.2f%%" % (accuracy * 100.0))

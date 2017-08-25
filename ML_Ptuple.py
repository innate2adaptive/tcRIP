# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 14:24:01 2017

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
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectFromModel

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
loadCDR12 = True
length = 14
tuplen = 2
secondaryClass= True # Take the feature importance vals  from Ada and run SVM

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
    
if squish:
    X=X[:50000]
    y=y[:50000]

# print class balances
dp.printClassBalance(y)

# 25% Validation set
xTrain, xVal, yTrain, yVal= train_test_split(X, y, test_size=0.20) 


#==============================================================================
# Adaboost
#==============================================================================

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

if secondaryClass:
    selec = SelectFromModel(clf, prefit=True, threshold=0.01)
    xTrain = selec.transform(xTrain)
    xVal = selec.transform(xVal)

    svm=SVC(C=10, gamma=0.01, class_weight='balanced',decision_function_shape='ovr')
    svm.fit(xTrain, yTrain)
    y_true, y_pred = yVal, svm.predict(xVal)
    accuracy = accuracy_score(y_true, y_pred)
    print("Validation Accuracy: %.2f%%" % (accuracy * 100.0))
    print(classification_report(yVal,y_pred))

#==============================================================================
# XGBoost
#==============================================================================
#print("======================================")
#print("Running Classification using XGBoost")
#
#model = XGBClassifier(reg_lambda=1)
#
#model.fit(xTrain, yTrain)
#
#y_pred = model.predict(xVal)
#predictions = [round(value) for value in y_pred]
#
#accuracy = accuracy_score(yVal, predictions)
#print("Validation Accuracy: %.2f%%" % (accuracy * 100.0))

#==============================================================================
# SVM
#==============================================================================
#print("======================================")
#print("Running Classification using SVM")
### grid search for best parameters for both linear and rbf kernels
###tuned_parameters = [{'kernel': ['rbf'], 'C': [0.1,1,10,100], 'gamma':[1e-3,1e-2,1e-1]}]
### runs grid search using the above parameter and doing 5-fold cross validation
###clf = GridSearchCV(SVC(C=1, gamma=0.01, class_weight='balanced',decision_function_shape='ovr'), tuned_parameters, cv=2, verbose=1)
###print(clf)
### Fit the svm
##
##
#clf=SVC(C=1, gamma=0.01, class_weight='balanced',decision_function_shape='ovr')
##n_estimators=10
###clf=BaggingClassifier(SVC(kernel='linear', class_weight='balanced'), max_samples=1.0 / n_estimators, n_estimators=n_estimators)
##
#
#clf.fit(xTrain, yTrain)
#
### Prints the cross validation report for the different parameters
##print("Best parameters set found on validation set:")
###print()
###print(clf.best_params_)
###print()
###print("Grid scores on development set:")
###print()
###means = clf.cv_results_['mean_test_score']
###stds = clf.cv_results_['std_test_score']
###for mean, std, params in zip(means, stds, clf.cv_results_['params']):
###    print("%0.3f (+/-%0.03f) for %r"
###          % (mean, std * 2, params))
###print()    
#    
#y_true, y_pred = yVal, clf.predict(xVal)
#
##print(classification_report(y_true, y_pred))
##
##print()    
#
#accuracy = accuracy_score(y_true, y_pred)
#print("Validation Accuracy: %.2f%%" % (accuracy * 100.0))

#==============================================================================
# NN
#==============================================================================

#def quickExpand(y):
#    newY=[]
#    for val in y:
#        small=[0,0]
#        small[int(val)]+=1
#        newY.append(small)
#    newY=np.array(newY)
#    return newY
#    
#y=quickExpand(y)
#
#
## 25% Validation set
#xTrain, xVal, yTrain, yVal= train_test_split(X, y, test_size=0.20) 
#
#
## defining the keras model
#inCDR3 = Input(shape=(xTrain.shape[1],))
#
##CDR1 pipeline
#layer1 = Dense(512, activation='relu')(inCDR3)
#layer2 = Dense(128, activation='relu')(layer1)
#layer3 = Dense(32, activation='relu')(layer2)
#predictions = Dense(2, activation='softmax')(layer3)
#
## define the model
#model = Model(inputs=inCDR3, outputs=[predictions])
#
#model.compile(optimizer='adam',
#              loss='categorical_crossentropy',
#              metrics=['accuracy'])
#
#
#model.fit(x=xTrain, 
#          y=yTrain, 
#          batch_size=256, 
#          epochs=10, 
#          verbose=1,
#          validation_split=0.2)
#
## evaluate the model
#scores = model.evaluate(xVal, yVal)
#print("\nTesting Accuracy: %.2f%%" % ( scores[1]*100))

#==============================================================================
# NN
#==============================================================================

#print("======================================")
#print("Running Classification using k-NN")
#
#model = KNN(n_neighbors=5)
#
#model.fit(xTrain, yTrain)
##print(model)   
#y_pred = model.predict(xVal)
#predictions = [round(value) for value in y_pred]
#
#accuracy = accuracy_score(yVal, predictions)
#print("Validation Accuracy: %.2f%%" % (accuracy * 100.0))

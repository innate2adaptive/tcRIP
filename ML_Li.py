# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 16:28:16 2017

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

from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier

from keras.models import Model
from keras.layers import Input, Dense
    
from sklearn.neighbors import KNeighborsClassifier as KNN

#==============================================================================
# ORGANIZE THE DATA
#==============================================================================

# load in sequences and vj info for all patients with naive and beta chains
seqs, vj = dp.loadAllPatients(['naive','beta'])

# filter out joint sequences
seqs[0], seqs[1], vj[0], vj[1], joint = dp.removeDup(seqs[0], seqs[1], vj[0], vj[1])

print("Number of Shared Seqs: {}".format(len(joint)))
print("Shared Percent: %.2f%%" % (len(joint)/(len(seqs[0])+len(seqs[1])) * 100.0))


#==============================================================================
# Parameters
#==============================================================================
squish = False
loadCDR12 = False
includeV= True
length=14
vlen=0
SM=True # Run SMOTE?
#==============================================================================
# Feat. Eng.
#==============================================================================

if loadCDR12:
    length=length+5+6
    # add extrac cds
    cddict=dp.extraCDs()
    seqs[0], vj[0]=dp.addCDextra(seqs[0],vj[0],cddict)
    seqs[1], vj[1]=dp.addCDextra(seqs[1],vj[1],cddict)

if includeV: 
    vlen+=1
    for idx, seqz in enumerate(seqs):
        seqs[idx]=dp.seq2fatch(seqz) 
        for idx2, seq in enumerate(seqs[idx]):
            new=np.zeros((1))
            try:
                new[0]=float(vj[idx][idx2])
            except:
                new[0]=float(vj[idx][idx2][0])
            seqs[idx][idx2]=np.concatenate((seq,new))


else:
    seqs[0]=dp.seq2fatch(seqs[0])
    seqs[1]=dp.seq2fatch(seqs[1])

seqs[0]=dp.filtr(seqs[0], length*5+vlen)
seqs[1]=dp.filtr(seqs[1], length*5+vlen)

# use function to create data
X, y = dp.dataCreator(seqs[0],seqs[1])

# comment this out if you dont want to focus on one CDR
#X=X[:,-1] # 95: is the CDR1, 70:95 is CDR2, :70 is cdr3
#X=np.expand_dims(X,1)

# shuffle data
X, y = sk.utils.shuffle(X,y)


   
if squish:
    X=X[:1000]
    y=y[:1000]

# print class balances
dp.printClassBalance(y)


# 25% Validation set
xTrain, xVal, yTrain, yVal= train_test_split(X, y, test_size=0.20) 

if SM:
   os=SMOTE() 
   xTrain, yTrain = os.fit_sample(xTrain,yTrain)


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
print(classification_report(yVal,y_pred))

y_true, y_pred = yVal, clf.predict_proba(xVal)

#==============================================================================
# XGBoost
#==============================================================================
#print("======================================")
#print("Running Classification using XGBoost")
#
#model = XGBClassifier(n_estimators=100,reg_lambda=1)
#
#model.fit(xTrain, yTrain)
##print(model)
#y_pred = model.predict(xVal)
#predictions = [round(value) for value in y_pred]
#
#accuracy = accuracy_score(yVal, predictions)
#print("Validation Accuracy: %.2f%%" % (accuracy * 100.0))
#print(classification_report(yVal,y_pred))
#
#



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
#clf.fit(xTrain, yTrain)
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
#y_true, y_pred = yVal, clf.predict(xVal)
##print(classification_report(y_true, y_pred))
##
##print()    
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
#
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

# -*- coding: utf-8 -*-
"""
This script takes the data from the first patient and analyzes it with sciKit 
learn. Takes data/patient1/... Tests have been done with both alpha and beta 
separetly and together.

"""
import numpy as np
import sklearn as sk
import getData as gt
import pdb
 
    
########################################################
# Data Pre Processing 
########################################################
X,Y=gt.getData(t="train", setup=2)


pdb.set_trace()

X.toarray()
Y=np.squeeze(Y)
# split into training, validation, test
# training 50% = :21267
# validation 25% = 21267:31901
# test 25% = 31901:
from sklearn.model_selection import train_test_split    
xTrain, xVal, yTrain, yVal = train_test_split(X, Y, test_size=0.75)    



########################################################
# Running SVM with RBF kernel
########################################################

# get and init the SVM using SciKitLearn
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# grid search for best parameters for both linear and rbf kernels
#tuned_parameters = [{'kernel': ['rbf'], 'C': [0.5,1,1.5]}] #{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
tuned_parameters = [{'kernel': ['rbf'], 'C': [100], 'gamma':[1e-3]}]

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
# reruns the classification on full size validation set then returns results
print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = yVal, clf.predict(xVal)
print(classification_report(y_true, y_pred))
print()



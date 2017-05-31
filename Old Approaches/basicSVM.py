# -*- coding: utf-8 -*-
"""
This script takes the data from the first patient and analyzes it with sciKit 
learn. Takes data/patient1/... Tests have been done with both alpha and beta 
separetly and together.

"""
import numpy as np
import atchFac as af
import sklearn as sk

### Switches
atchConversion=False

cd4a=np.delete(np.load('data/cd4A.npy'),1,1)
cd8a=np.delete(np.load('data/cd8A.npy'),1,1)
cd4b=np.delete(np.load('data/cd4B.npy'),1,1)
cd8b=np.delete(np.load('data/cd8B.npy'),1,1)



if atchConversion==True:

    #acd4a=np.zeros((len(cd4a),1))
    #acd8a=np.zeros((len(cd8a),1))
    #acd4b=np.zeros((len(cd4b),1))
    #acd8b=np.zeros((len(cd8b),1))
    
    acd4a=list()
    acd8a=list()
    acd4b=list()
    acd8b=list()

    # list of lists for better clarity
    cd=[cd4a,cd8a,cd4b,cd8b]
    acd=[acd4a,acd8a,acd4b,acd8b]

    # As per lengthHistogram.py the most common length is 14
    # filtering each vector to have only length 14
    
    # length to do split 
    splitlen=14
    
    # first creating matrix of atchely factors
    for idx, c in enumerate(cd):
        # go through each of the four sets of values
        for ch in c:
            # go through each row 
            # if row size is 14 then do the conversion
            if len(ch[0])==splitlen:
                # go through  each letter and get the corresponding matrix values
                tempAr=np.zeros((splitlen,5))
                for i in range(splitlen):
                    # using the actchley functino in atchFac.py
                    AA=ch[0][i]
                    tempAr[i,:]=af.atchleyFactor(AA)[:]
                # append new 14x5 data point to list of lists
                acd[idx].append(tempAr)
            
    np.save('data/acd4A',acd[0])
    np.save('data/acd8A',acd[1])
    np.save('data/acd4B',acd[2])
    np.save('data/acd8B',acd[3])
        
else:
        
    acd4a=np.load('data/acd4A.npy')
    acd8a=np.load('data/acd8A.npy')
    acd4b=np.load('data/acd4B.npy')
    acd8b=np.load('data/acd8B.npy')

    
    
########################################################
# Data Pre Processing 
########################################################

# set random seeds
np.random.seed(80085)

# combine alpha and beta chains
#acd4=np.concatenate((acd4a,acd4b),0)
#acd8=np.concatenate((acd8a,acd8b),0)
acd4=acd4a
acd8=acd8a
acd4a=None
acd4b=None
acd8a=None
acd8b=None

# create vectors of labels
# label for cd4 is 4 and vice versa
acd4Label=np.zeros((len(acd4)))
acd8Label=np.zeros((len(acd8)))
acd4Label[:]=4
acd8Label[:]=8

# combine labels and data
data=np.concatenate((acd4,acd8),0)
labels=np.concatenate((acd4Label,acd8Label),0)

# shuffle using sklearn which conserves order
data, labels = sk.utils.shuffle(data,labels)
data = data.reshape(data.shape[0],-1)


# split into training, validation, test
# training 50% = :21267
# validation 25% = 21267:31901
# test 25% = 31901:
from sklearn.model_selection import train_test_split    
xTrain, xHalf, yTrain, yHalf = train_test_split(data, labels, test_size=0.60)    
xVal, xTest, yVal, yTest = train_test_split(xHalf, yHalf, test_size=0.50) 

#xTrain=data[:21267,:,:].reshape(21267,-1)
#xVal=data[21267:31901,:,:].reshape(10634,-1)
#xTest=data[31901:,:,:].reshape(10633,-1)

#yTrain=labels[:21267]
#yVal=labels[21267:31901]
#yTest=labels[31901:]


########################################################
# Running SVM with RBF kernel
########################################################

# get and init the SVM using SciKitLearn
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# grid search for best parameters for both linear and rbf kernels
#tuned_parameters = [{'kernel': ['rbf'], 'C': [0.5,1,1.5]}] #{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
tuned_parameters = [{'kernel': ['rbf'], 'C': [100,1000], 'gamma':[1e-3]}]

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



# -*- coding: utf-8 -*-
"""
This script takes the data from the first patient and analyzes it with sciKit 
learn. Takes data/patient1/... Tests have been done with both alpha and beta 
separetly and together.

"""
import numpy as np
import atchFac as af
import sklearn as sk
import pdb
from sklearn.cluster import MiniBatchKMeans

np.random.seed(80085)

### Switches
atchConversion=False # run conversion for sequences
pTupleSplit=False    # calculate p-tuples as save
loadTuple=True       # load p-tuples saved
kmean=True           # calculate k-means

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
        
"""else:
        
    acd4a=np.load('data/acd4A.npy')
    acd8a=np.load('data/acd8A.npy')
    acd4b=np.load('data/acd4B.npy')
    acd8b=np.load('data/acd8B.npy')"""
    
def pTuple(vec,n=3):
    n=3
    return [vec[i:i+n] for i in range(len(vec)-n+1)]
    

if pTupleSplit==True:
    
    pcd4a=list()
    pcd8a=list()
    pcd4b=list()
    pcd8b=list()

    # list of lists for better clarity
    cd=[cd4a,cd8a,cd4b,cd8b]
    acd=[pcd4a,pcd8a,pcd4b,pcd8b]
    
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
                chVec=pTuple(ch[0])
                chAVec=[]
                # now have list of 3 character values
                for tup in chVec:
                    tempAr=np.zeros((3,5))
                    for i in range(3):
                        # using the actchley functino in atchFac.py
                        AA=tup[i]
                        tempAr[i,:]=af.atchleyFactor(AA)[:]
                    chAVec.append(tempAr.reshape(15))
                # append new 14x5 data point to list of lists
                acd[idx].append(chAVec)
            
    np.save('data/pcd4A',acd[0])
    np.save('data/pcd8A',acd[1])
    np.save('data/pcd4B',acd[2])
    np.save('data/pcd8B',acd[3])

if loadTuple==True:
    print("Loading List of p-tuples")
    pcd4a=np.load('data/pcd4A.npy')
    pcd8a=np.load('data/pcd8A.npy')
    pcd4b=np.load('data/pcd4B.npy')
    pcd8b=np.load('data/pcd8B.npy')


    
########################################################
# K-Means data processing
########################################################
if kmean==True:
    from collections import Counter
    
    tupleArray=np.concatenate((pcd4a,pcd4b,pcd8a,pcd8b),axis=0)    
    premean=tupleArray.reshape(-1,15)
    # now have massive list of 3-tuples 
    # will shuffle and truncate to 10000 to have samples
    
    premean=np.asarray(premean)
    np.random.shuffle(premean)
    premean=premean[:10000]
    print('running kMeans...')
    # run kmeans using sci kit learn
    kmeans = MiniBatchKMeans(n_clusters=100, verbose=0).fit(premean)
    #cluster=kmeans.cluster_centers_ # 100, 15 
    
    # cluster contains the values that will now be used to computer the feature dictionary
    # get proportions of q=10000 in each cluster
    props=Counter(kmeans.labels_)
    # normalize so there are proportions for each of the 100 spaces
    total = sum(props.values(), 0.0)
    for key in props:
        props[key] /= total
    
    
########################################################
# Feature function
########################################################


# define a feature function
def event_feat(event):
    """
    This feature function returns a dictionary representation of the sequence. 
    Args:
        event: sequence of values
    Returns:
        a feature vector
    """
    
    #Once all vectors are allocated, the
    #number of vectors within each cluster is counted and converted into
    #a proportion of the total number of Atchley vectors selected 
    
    assigned=kmeans.predict(event)
    assCount=Counter(assigned)
    result=np.zeros((100))
    for i in range(100):
        result[i]=assCount[i] #*props[i]
    
    return result

# create vectors of labels
# label for cd4 is 4 and vice versa
acd4Label=np.zeros((len(pcd4a)+len(pcd4b)))
acd8Label=np.zeros((len(pcd8a)+len(pcd8b)))
acd4Label[:]=4
acd8Label[:]=8
labels=np.concatenate((acd4Label,acd8Label),0)      
    
event_train=tupleArray # clearer notation, this variable comes from kmean calc.

event_train, labels = sk.utils.shuffle(event_train,labels)

print('running feature function...')
## We convert the event candidates and their labels into vectors and integers, respectively.
train_event_x = [event_feat(x) for x in event_train]
data=train_event_x # shifting names for consistency
data=np.asarray(data)    

#pdb.set_trace()
########################################################
# Data Pre Processing 
########################################################

# split into training, validation, test
# training 75% 
# validation 25% 
# doing cross validation
from sklearn.model_selection import train_test_split    
xTrain, xVal, yTrain, yVal = train_test_split(data, labels, test_size=0.75)



########################################################
# Running SVM with RBF kernel
########################################################

# get and init the SVM using SciKitLearn
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# grid search for best parameters for both linear and rbf kernels
tuned_parameters = [{'kernel': ['rbf'], 'C': [10,100]}] #{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}

print("Running Grid Search With Cross Validation for SVM")
# runs grid search using the above parameter and doing 5-fold cross validation
clf = GridSearchCV(SVC(C=1, class_weight='balanced'), tuned_parameters, cv=2, verbose=1)
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



# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 11:31:43 2017

@author: lewismoffat


This builds a classifier for based on positional probabilities

"""


########################################################
# Module Imports 
########################################################
import numpy as np
import dataProcessing as dp
import pdb
import sklearn as sk
import matplotlib.pyplot as plt
import pandas
from collections import Counter
from operator import itemgetter
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier as KNN
from matplotlib import pylab
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


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

length=14
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
            
            # Here filtering is put in place to just pull out the X long amino 
            # acids e.g. 14 aa long
            if len(twoVals[1][0])==length:
                seqs[index].append(twoVals[1][0])
                vj[index].append(twoVals[0][:2])
            
# data is in the format [v_index, j_index, deletionsV, deletionsJ, extra:CDR3, count]

# vj contains the vj index locations
# seqs contains the sequence locations




########################################################
# Frequency Analysis - Positions
########################################################
"""
This is split by class as usual. It will be built into a function for ease
"""
# This is used quite often so laying it here
intDictzero={ 'A': 0 ,
              'C': 0 ,
              'D': 0 ,
              'E': 0 ,
              'F': 0 ,
              'G': 0 ,
              'H': 0 ,
              'I': 0 ,
              'K': 0 ,
              'L': 0 ,
              'M': 0 ,
              'N': 0 ,
              'P': 0 ,
              'Q': 0 ,
              'R': 0 ,
              'S': 0 ,
              'T': 0 ,
              'V': 0 ,
              'W': 0 ,
              'Y': 0 ,}

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

    
# its handy to have the look up dictionaries for 
cd4Prob, dl4=positionProb(cd4, length, intDictzero)
cd8Prob, dl8=positionProb(cd8, length, intDictzero)

pos=12

plt.imshow(np.expand_dims(cd4Prob.T[:,pos],1))
#plt.imshow(cd4Prob.T)
plt.title('CD4 Heat Map of AA usage for 13 long AAs')
plt.xlabel('AA Position')
plt.ylabel('Amino Acid')
plt.yticks(np.arange(20),['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'])
plt.show()

plt.imshow(np.expand_dims(cd8Prob.T[:,pos],1))
#plt.imshow(cd8Prob.T)
plt.title('CD8 Heat Map of AA usage for 13 long AAs')
plt.xlabel('AA Position')
plt.ylabel('Amino Acid')
plt.yticks(np.arange(20),['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'])
plt.show()



plt.imshow(np.absolute(np.expand_dims(cd4Prob.T[:,pos],1)-np.expand_dims(cd8Prob.T[:,pos],1)))
#plt.imshow(cd4Prob.T)
plt.title('Heat Map of Difference in AA usage for 13 long AAs')
plt.xlabel('AA Position')
plt.ylabel('Amino Acid')
plt.yticks(np.arange(20),['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'])
plt.show()


########################################################
# Frequency Analysis - Prior Probabilty of V usage
########################################################


########################################################
# Frequency Analysis - Feature Generation
########################################################

# now that the dict is done we canuse it to construct the feature vectors for
# the sequences 
run=[]
run8=[]
for idx, char in enumerate(cd4[100]):
    run.append(dl4[idx][char])
    run8.append(dl8[idx][char])
print(np.log(np.product(run)))
print(np.log(np.product(run8)))


########################################################
# Classification nusing K-NN
########################################################

## funtion that replaces each sequence with its probability position values
#def charForProb(seqs,dic4, dic8):
#    newSeqs=[]
#    for seq in seqs:
#        run=[]
#        run2=[]
#        for idx, char in enumerate(seq):
#            run.append(dic4[idx][char])
#            run2.append(dic8[idx][char])
#        run=np.log(np.product(run)+1)
#        run2=np.log(np.product(run2)+1)
#        newSeqs.append([run, run2])
#    return newSeqs
#
#
#
#cd4=charForProb(cd4,dl4,dl8)
#cd8=charForProb(cd8,dl4,dl8)
#
#
## from this point it is assumed cd4/8 are NUMPY vectors
## labels are created and then the X and Y vectors are shuffled and combo'd
#y4=np.zeros((len(cd4)))
#y8=np.zeros((len(cd8)))
#y4[:]=0
#y8[:]=1
#
## combine classes
#Y = np.concatenate((y4,y8),0)
#X = np.concatenate((cd4,cd8),0)
#
#print("CD4 to CD8 Ratio {}:{}".format(len(cd4)/(len(cd4)+len(cd8)),len(cd8)/(len(cd4)+len(cd8))))
#print("Total Sequences: {}".format(len(cd4)+len(cd8)))
## memory clean up
##cd4=None
##cd8=None
#y4=None
#y8=None
#
#
#X, Y = sk.utils.shuffle(X,Y)
#xTrain, xHalf, yTrain, yHalf = train_test_split(X, Y, test_size=0.20) 
## memory clean up
#X=None
#Y=None 
#
#xVal, xTest, yVal, yTest= train_test_split(xHalf, yHalf, test_size=0.50) 
#
## Some memory clean up
#xHalf=None
#yHalf=None
#sqHalf=None
#
#
#print("Data Loaded and Ready...")
#print("Training K-NN")
#neigh=KNN(n_neighbors=1)
#neigh.fit(xTrain, yTrain)
#y_true, y_pred = yVal, neigh.predict(xVal)
#print("{} Validaton Accuracy".format(accuracy_score(y_true, y_pred)))
#print(classification_report(y_true, y_pred))
#
#
#
## grid search for best parameters for both linear and rbf kernels
#tuned_parameters = [{'kernel': ['rbf'], 'C': [10], 'gamma':[1e-3]}]
## runs grid search using the above parameter and doing 5-fold cross validation
#clf = GridSearchCV(SVC(C=1, class_weight='balanced',decision_function_shape='ovr'), tuned_parameters, cv=2, verbose=1)
#print(clf)
## Fit the svm
#clf.fit(xTrain, yTrain)
## Prints the cross validation report for the different parameters
#print("Best parameters set found on validation set:")
#print()
#print(clf.best_params_)
#print()
#print("Grid scores on development set:")
#print()
#means = clf.cv_results_['mean_test_score']
#stds = clf.cv_results_['std_test_score']
#for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#    print("%0.3f (+/-%0.03f) for %r"
#          % (mean, std * 2, params))
#print()        
#y_true, y_pred = yVal, clf.predict(xVal)
#print(classification_report(y_true, y_pred))
#print()
#
#
#
#
##print("Running tSNE")
##        
##tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
##two_d_embeddings_1 = tsne.fit_transform(xTrain[:10000])
##
##def plot(embeddings, labels):
##    assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
##    pylab.figure(figsize=(15,15))  # in inches
##    for i, label in enumerate(labels):
##        x, y = embeddings[i,:]
##
##        if label>0.5:
##            pylab.scatter(x, y, c='b')
##        else:
##            pylab.scatter(x, y, c='r')
##        #pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',  ha='right', va='bottom')
##    pylab.show()
##    return
##    
##plot(two_d_embeddings_1, yTrain[:10000])
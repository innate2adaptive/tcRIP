# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 14:25:08 2017

@author: lewismoffat


This script is statistics focused. It generates the general plots and values
for the V/J gene usage statistics.

It is expected that the data is located in the directory: F:/seqs/*.cdr

"""



#==============================================================================
# Module Imports 
#==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import dataProcessing as dp
import pdb
import seaborn as sns 
import sklearn as sk

from collections import defaultdict
from collections import Counter

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from matplotlib import pylab


from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE


#==============================================================================
# Get the data
#==============================================================================

# are we doing the full set
singlePatient=False

# which patient to get data from 
patient=['Complete']
chain = "beta"


if singlePatient:
    print('Patient: '+patient[0])
    delim = ["naive",chain]+patient #other delimiters
else:
    print('Patient: All')
    delim = ["naive",chain] #other delimiters


seqs, vj = dp.loadAllPatients(delim) # these gets all the sequences and vj values

#==============================================================================
# Basic Analytics
#==============================================================================
runhisto=False
if runhisto:
    # filter out joint sequences
    seqs[0], seqs[1], vj[0], vj[1], joint = dp.removeDup(seqs[0], seqs[1], vj[0], vj[1])
    
    # use function to create data
    X, y = dp.dataCreator(seqs[0],seqs[1])
    
    vj[0]=np.array([list(map(int, x)) for x in vj[0]]) # make them all ints!
    vj[1]=np.array([list(map(int, x)) for x in vj[1]]) # make them all ints!
    
    
    # print the J gene usage CD4
    dp.standardHisto(vj[0][:,1], Title="CD4 J Gene Usage", xlabel="J Gene Usage by Index", size=[5,5])
    dp.standardHisto(vj[1][:,1], Title="CD8 J Gene Usage", xlabel="J Gene Usage by Index", size=[5,5])
    
    dp.standardHistoV2(vj[0][:,1],vj[1][:,1], Title="J Gene Usage", xlabel="J Gene Usage by Index", size=[5,5])
    
    # print the V Gene usage CD8
    dp.standardHisto(vj[0][:,0], Title="CD4 V Gene Usage", xlabel="V Gene Usage by Index", size=[15,5])
    dp.standardHisto(vj[1][:,0], Title="CD8 V Gene Usage", xlabel="V Gene Usage by Index", size=[15,5])

    dp.standardHistoV2(vj[0][:,0],vj[1][:,0], Title="V Gene Usage", xlabel="V Gene Usage by Index", size=[15,5])
    
#==============================================================================
# Analysis of V-J region being used
#==============================================================================

# this takes the data invariantly into a dict and then records info
# after this the data is separated to 14 long seqs of the most common 
# vj combination which is v=15, j=12

seq_mapper=dp.dictEncoder(seqs, vj, filtr=True, clip=False, filtrLen=14)
# the data is now stored in a dictionary and the keys are the X
# need to convert the data to one hot encoded
X_char=list(seq_mapper.keys()) # get the seqs 

# get the vjs amd convert them to strings which are put in a counter
vjs=[str(x[1][0])+','+str(x[1][1]) for x in list(seq_mapper.values())]

# The most common cluster with 3874 examples is 15,12
vjc=Counter(vjs)
newCD4s=[]
newCD8s=[]
for key in list(seq_mapper.keys()):
    row=seq_mapper[key]
    try:
        vandj=row[1].tolist() # sometimes it comes out as a numpy vector
    except:
        vandj=row[1]
    if vandj==[15,12]:     # this is the most common combination
        if row[0]=="CD4":
            newCD4s.append(key)
        else:
            newCD8s.append(key)

# convert to atchley method for visualization

newCD4=dp.seq2fatch(newCD4s.copy())
newCD8=dp.seq2fatch(newCD8s.copy())

# use function to create data
X, y = dp.dataCreator(newCD4,newCD8)


#==============================================================================
# Run tSNE 
#==============================================================================
runVJtsne=False
if runVJtsne:
    print("Running tSNE")       
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, verbose=1)
    two_d_embeddings_1 = tsne.fit_transform(X)
    
    def plot(embeddings, labels, seqs):
        assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
        pylab.figure(figsize=(100,100))  # in inches
        for i, label in enumerate(labels):
            x, y = embeddings[i,:]
            if label>0.5:
                pylab.scatter(x, y, c='b')
            else:
                pylab.scatter(x, y, c='r')
            # comment in for labels as the seqs to be added
            pylab.annotate(seqs[i], xy=(x, y), xytext=(5, 2), textcoords='offset points',  ha='right', va='bottom')
        pylab.title("t-SNE 2-D Embeddings of Most Common V-J Combination")
        pylab.show()
        return
    # comment in to produce the maaaaasive plot  
    #plot(two_d_embeddings_1, y, newCD4s+newCD4s)
    
    # plot using seaborn to get a heatmap, looks pretty in two different graphs
    
    sns.jointplot(x=two_d_embeddings_1[:,0], y=two_d_embeddings_1[:,1], kind='kde', maringal_kws=dict(n_levels=30))
    plt.show()

    
    # now cluster them using kmeans
    km=KMeans(n_clusters=30)
    km.fit(two_d_embeddings_1)
    labels=km.labels_ # nifty for colors
    plt.figure(figsize=(5,5))
    plt.title("t-SNE 2-D Embeddings of Most Common V-J Combination Clustered with K-Means")
    plt.scatter(two_d_embeddings_1[:,0],two_d_embeddings_1[:,1],c=labels.astype(np.float))
    plt.show()
    
#==============================================================================
# Repeat for 10k examples of all seqs
#==============================================================================
runTsne=False
if runTsne:
    seqs[0]=dp.filtr(seqs[0], 14)
    seqs[1]=dp.filtr(seqs[1], 14)
    
    newCD4=dp.seq2fatch(seqs[0][:5000])
    newCD8=dp.seq2fatch(seqs[1][:5000])
    
    # use function to create data
    X, y = dp.dataCreator(newCD4,newCD8)
    
    
    print("Running tSNE")       
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, verbose=1)
    two_d_embeddings_1 = tsne.fit_transform(X)
    
    def plot(embeddings, labels, seqs):
        assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
        pylab.figure(figsize=(10,10))  # in inches
        for i, label in enumerate(labels):
            x, y = embeddings[i,:]
            if label>0.5:
                pylab.scatter(x, y, c='b')
            else:
                pylab.scatter(x, y, c='r')
            # comment in for labels as the seqs to be added
            #pylab.annotate(seqs[i], xy=(x, y), xytext=(5, 2), textcoords='offset points',  ha='right', va='bottom')
        pylab.title("t-SNE 2-D Embeddings of 10k examples from Complete Dataset")
        pylab.show()
        return
        
    plot(two_d_embeddings_1, y, seqs[0][:5000]+seqs[1][:5000])
    
    # plot using seaborn to get a heatmap, looks pretty in two different graphs
    
    sns.jointplot(x=two_d_embeddings_1[:,0], y=two_d_embeddings_1[:,1], kind='kde', maringal_kws=dict(n_levels=30))
    plt.show()
  
    
#==============================================================================
#  Run classification on Just V and J 
#==============================================================================
    
# use function to create data
X, y = dp.dataCreator(vj[0],vj[1])

# shuffle data
X, y = sk.utils.shuffle(X,y)

# print class balances
dp.printClassBalance(y)

# 20% Validation set
xTrain, xVal, yTrain, yVal= train_test_split(X, y, test_size=0.20) 
    
print("======================================")
print("Running Classification using XGBoost")

model = XGBClassifier(n_estimators=100,reg_lambda=1)
os=SMOTE() 
xTrain, yTrain = os.fit_sample(xTrain,yTrain)
model.fit(xTrain, yTrain)
y_pred = model.predict(xVal)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(yVal, predictions)
print("Validation Accuracy: %.2f%%" % (accuracy * 100.0))

# now do for just v
X_new=np.expand_dims(X[:,0],1)
# 20% Validation set
xTrain, xVal, yTrain, yVal= train_test_split(X_new, y, test_size=0.20) 
os=SMOTE() 
xTrain, yTrain = os.fit_sample(xTrain,yTrain)
model.fit(xTrain, yTrain)
#print(model)
y_pred = model.predict(xVal)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(yVal, predictions)
print("Validation Accuracy: %.2f%%" % (accuracy * 100.0))

# now do for just j
X_new=np.expand_dims(X[:,1],1)
# 20% Validation set
xTrain, xVal, yTrain, yVal= train_test_split(X_new, y, test_size=0.20) 
os=SMOTE() 
xTrain, yTrain = os.fit_sample(xTrain,yTrain)
model.fit(xTrain, yTrain)
#print(model)
y_pred = model.predict(xVal)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(yVal, predictions)
print("Validation Accuracy: %.2f%%" % (accuracy * 100.0))



    
    

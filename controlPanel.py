# -*- coding: utf-8 -*-
"""
Created on Thu May 25 09:46:57 2017

@author: lewismoffat
"""

########################################################
# Module Imports 
########################################################
import numpy as np
import dataProcessing as dp
import rnnModel as r2n
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
from matplotlib import pylab
########################################################
# Parameters
########################################################
# big parameters
train       = True
RNN         = False
pureNN      = False
onlyCD3     = True # this means files contain CDR1/2 sequences so have to be parsed differently
autoEncoder = False
svm         = True
tSNE        = False
knn         = True
NBayes      = False

# module parameters in dictionaries
dataParams={'integerize':False,
            'clipping':False,
            'clipLen':12, # this clips sequences to this length
            'filter':True, # this filters only sequences of this length
            'filterLen':14, #
            'pTuple':False,
            'pTuplelen':2,
            'kmeans':False,
            'originalAtch':True,
            'GloVe':False,
            'window':True
            }

ControllerParams={'batch_size':32,
                     'epochs':10}
                     
aeControllerParams={'batch_size':128,
                     'epochs':500,
                     'learningRate':0.001}
                     
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
            
# File names for different data; A - alpha chain, B - beta chain            
cd4A_file = 'patient1/vDCRe_alpha_EG10_CD4_naive_alpha.txt'
cd8A_file = 'patient1/vDCRe_alpha_EG10_CD8_naive_alpha.txt'
cd4B_file = 'patient1/vDCRe_beta_EG10_CD4_naive_beta.txt'
cd8B_file = 'patient1/vDCRe_beta_EG10_CD8_naive_beta.txt'
data = 'data/'
extra = 'extra/'
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
seqs=[cd4,cd8]

# if the data files you want only have cdr3s it has to parse it differently
if onlyCD3:
    for index, file in enumerate(files):
        file=data+file
        with open(file,'r') as infile:
            # goes through each of the files specified in read mode and pulls out 
            # each line and formats it so a list gets X copies of the sequence 
            for line in infile:
                twoVals=line.split(", ")
                twoVals[1]=twoVals[1].replace("\n","")
                for i in range(int(twoVals[1])):
                    seqs[index].append(twoVals[0])
else:
    for index, file in enumerate(files):
        file=data+extra+file
        with open(file,'r') as infile:
            # goes through each of the files specified in read mode and pulls out 
            # each line adds each sequence from the line 
            for line in infile:
                threeVals=line.split(",")
                threeVals[2]=threeVals[2].replace("\n","")
                for i in threeVals:
                    if i=="":
                        continue
                    seqs[index].append(i)



    




# at this point both cd4 and cd8 are filled with lists of sequences as strings
# counts have been factored in

# This short code block prints a graph with a bar chart histogram of the 
# length distribution for both CD4 and CD8 sequences together.
# Flip the boolean to enable/disable it
lenHist=False
if lenHist:
    # length histogram
    lenArr=[]
    # create new vector of lengths 
    for seq in cd4+cd8:
        lenArr.append(len(seq))
        
    # convert list to numpy array
    lenArr=np.asarray(lenArr)
    
    # bins
    binNum=max(lenArr)-min(lenArr)
    bins=np.arange(min(lenArr),max(lenArr))
    plt.hist(lenArr, bins=binNum)
    plt.xticks(bins)
    plt.title("Sequence Length Histogram")
    plt.xlabel("Sequence Length (AA)")
    plt.ylabel("Frequency")
    plt.show()

########################################################
# Data Processing
########################################################
"""
The following processes the data based on what you have specified in the dicts
Current setup is just for replacing characters with integer IDs
"""
# find the length of the longest amino acid 
longest = len(max(seqs[0], key=len))
longest1 = len(max(seqs[1], key=len))
longest = max([longest,longest1])
ModelParams['maxLen']=longest

# calculate sequence lengths for tf models
sq4=[len(x) for x in cd4]
sq8=[len(x) for x in cd8]
sqlen=np.concatenate((sq4,sq8))

# this will clip the maximum length of the sequence
if dataParams['clipping']==True:
    cd4=dp.clip(cd4, dataParams['clipLen'])
    cd8=dp.clip(cd8, dataParams['clipLen'])
    for idx, sq in enumerate(sqlen):
        if sq>dataParams['clipLen']:
            sqlen[idx]=dataParams['clipLen']

     
# this will return only sequences of a set length
if dataParams['filter']==True:
    cd4=dp.filtr(cd4, dataParams['filterLen'])
    cd8=dp.filtr(cd8, dataParams['filterLen'])
    sqlen=[]
    for i in range(len(cd4)+len(cd8)):
        sqlen.append(dataParams['filterLen'])
    ModelParams['maxLen']=dataParams['filterLen']

# removes the first four AAs
if dataParams['window']==True:
    for idx, sq in enumerate(cd4):
        cd4[idx]=sq[2:10]
    for idx, sq in enumerate(cd8):
        cd8[idx]=sq[2:10]


#cd4=dp.expandTuples(cd4, 6)
#cd8=dp.expandTuples(cd8, 6)
#sq4=[len(x) for x in cd4]
#sq8=[len(x) for x in cd8]
#sqlen=np.concatenate((sq4,sq8))


# this will replace amino acids with an integer ID specified in dataProcessing.py
if dataParams['integerize']==True:
    cd4=dp.char2int(cd4, longest)
    cd8=dp.char2int(cd8, longest)
    seqs=None

    
if dataParams['originalAtch']==True:
    cd4=dp.seq2fatch(cd4)
    cd8=dp.seq2fatch(cd8)
    
if dataParams['GloVe']==True:
    cd4=dp.GloVe(cd4, False)
    cd8=dp.GloVe(cd8, False)
    
# from this point it is assumed cd4/8 are NUMPY vectors
# labels are created and then the X and Y vectors are shuffled and combo'd
y4=np.zeros((len(cd4)))
y8=np.zeros((len(cd8)))
y4[:]=0
y8[:]=1

# combine classes
Y = np.concatenate((y4,y8),0)
X = np.concatenate((cd4,cd8),0)

print("CD4 to CD8 Ratio {}:{}".format(len(cd4)/(len(cd4)+len(cd8)),len(cd8)/(len(cd4)+len(cd8))))
print("Total Sequences: {}".format(len(cd4)+len(cd8)))
# memory clean up
cd4=None
cd8=None
y4=None
y8=None
twoVals=None
sq4=None
sq8=None

# creates pTuples of length three if wanted
if dataParams['pTuple']==True:
    if dataParams['kmeans']==True:
        X=dp.kmeans(X, n=dataParams['pTuplelen'])
    else:
        X=dp.char2ptuple(X, n=dataParams['pTuplelen'])
        ModelParams['maxLen']=X.shape[1]

# shuffle data
X, Y, sqlen = sk.utils.shuffle(X,Y,sqlen)

shorten=False
if shorten:
    X=X[:1000]
    Y=Y[:1000]
    sqlen=sqlen[:1000]


if RNN==True and ModelParams['embed']==False:
    X=np.reshape(X,(-1,5,dataParams['filterLen']))
    
xTrain, xHalf, yTrain, yHalf, sqTrain, sqHalf = train_test_split(X, Y, sqlen, test_size=0.20) 
# memory clean up
X=None
Y=None 

xVal, xTest, yVal, yTest, sqVal, sqTest = train_test_split(xHalf, yHalf, sqHalf, test_size=0.50) 

# Some memory clean up
xHalf=None
yHalf=None
sqHalf=None


print("Data Loaded and Ready...")
    
########################################################
# Model Setup 
########################################################
if train==True:
    if autoEncoder==True:
        # Spool up warp drives! This gets the rnn controller class going
        aeMain = ae.Controller(aeControllerParams, ModelParams) 
        print("Training AE")
        xTrain, xVal = aeMain.train(xTrain, yTrain, sqTrain, xVal, yVal, sqVal)
    if RNN==True:
        # Spool up warp drives! This gets the rnn controller class going
        rnnMain = r2n.Controller(ControllerParams, ModelParams) 
        print("Training RNN")
        rnnMain.train(xTrain, yTrain, sqTrain, xVal, yVal, sqVal)
    if pureNN==True:
        # Spool up warp drives! This gets the normal nn controller class going
        ModelParams['maxLen']=xTrain.shape[1]
        nnMain=n2.Controller(ControllerParams, ModelParams) 
        print("Training NN")
        nnMain.train(xTrain, yTrain, sqTrain, xVal, yVal, sqVal)
    if svm==True:
        # grid search for best parameters for both linear and rbf kernels
        tuned_parameters = [{'kernel': ['rbf'], 'C': [10], 'gamma':[1e-3]}]
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
    
    if knn:
        print("Training K-NN")
        neigh=KNN(n_neighbors=1)
        neigh.fit(xTrain, yTrain)
        y_true, y_pred = yVal, neigh.predict(xVal)
        print("{} Validaton Accuracy".format(accuracy_score(y_true, y_pred)))
        print(classification_report(y_true, y_pred))
        
    if tSNE==True:
        print("Running tSNE")
        
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        two_d_embeddings_1 = tsne.fit_transform(xTrain[:10000])
        
        def plot(embeddings, labels):
            assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
            pylab.figure(figsize=(15,15))  # in inches
            for i, label in enumerate(labels):
                x, y = embeddings[i,:]

                if label>0.5:
                    pylab.scatter(x, y, c='b')
                else:
                    pylab.scatter(x, y, c='r')
                #pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',  ha='right', va='bottom')
            pylab.show()
            return
            
        plot(two_d_embeddings_1, yTrain[:10000])
#        pylab.figure(figsize=(15,15)) 
#        for i, label in enumerate(xTrain):
#            x, y = two_d_embeddings_1[i,:]
#            
#            pylab.scatter(x, y, c='b')
#        pylab.show()
#        
        
        

        
        
        
        
        




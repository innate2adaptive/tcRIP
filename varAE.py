# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 12:37:40 2017

@author: Lewis
"""

########################################################
# Module Imports 
########################################################
import numpy as np
import dataProcessing as dp
import rnnModel as r2n
import pureModel as n2
import pdb
import sklearn as sk
from sklearn.model_selection import train_test_split 
import AEModel as ae


########################################################
# Parameters
########################################################
# big parameters
train=True
RNN = False
pureNN = False
onlyCD3 = True # this means files contain CDR1/2 sequences so have to be parsed differently
varAE=True

# module parameters in dictionaries
dataParams={'integerize':False,
            'clipping':False,
            'clipLen':14, # this clips sequences to this length
            'filter':True, # this filters only sequences of this length
            'filterLen':14, #
            'pTuple':False,
            'kmeans':False,
            'originalAtch':True
            }
rnnControllerParams={'batch_size':128,
                     'epochs':10000}
                     
rnnModelParams={'learningRate':0.001,
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
                'batch_size':rnnControllerParams['batch_size'],
                'conv':False,
                'embed':False,
                'save':False,
                'load':True}
            
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
files = [cd4A_file, cd8A_file]

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
rnnModelParams['maxLen']=longest

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
    rnnModelParams['maxLen']=dataParams['filterLen']

# this will replace amino acids with an integer ID specified in dataProcessing.py
if dataParams['integerize']==True:
    cd4=dp.char2int(cd4, longest)
    cd8=dp.char2int(cd8, longest)
    seqs=None

    
if dataParams['originalAtch']==True:
    cd4=dp.seq2fatch(cd4)
    cd8=dp.seq2fatch(cd8)
    
    
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
        X=dp.kmeans(X)
    else:
        X=dp.char2ptuple(X)

# shuffle data
X, Y, sqlen = sk.utils.shuffle(X,Y,sqlen)

if RNN==True and rnnModelParams['embed']==False:
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
    if varAE==True:
        aeMain = ae.Controller(rnnControllerParams, rnnModelParams)
        print("Training RNN")
        aeMain.train(xTrain, yTrain, sqTrain, xVal, yVal, sqVal)
        
            
    if RNN==True:
        # Spool up warp drives! This gets the rnn controller class going
        rnnMain = r2n.Controller(rnnControllerParams, rnnModelParams) 
        print("Training RNN")
        rnnMain.train(xTrain, yTrain, sqTrain, xVal, yVal, sqVal)
    if pureNN==True:
        # Spool up warp drives! This gets the normal nn controller class going
        rnnModelParams['maxLen']=xTrain.shape[1]
        nnMain=n2.Controller(rnnControllerParams, rnnModelParams) 
        print("Training NN")
        nnMain.train(xTrain, yTrain, sqTrain, xVal, yVal, sqVal)
        
        












      
        
        

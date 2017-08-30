# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 12:21:10 2017

@author: lewismoffat

This script runs a AutoEncoder network on all the data for one-hot
encoded amino acids
"""

#==============================================================================
# IMPORTS
#==============================================================================
import tensorflow as tf
import numpy as np

import dataProcessing as dp
import sklearn as sk
import pdb

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier

from imblearn.over_sampling import SMOTE

from tensorflow.contrib import slim

from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier as KNN

#==============================================================================
# ORGANIZE THE DATA
#==============================================================================

# load in sequences and vj info for all patients with naive and beta chains
seqs, vj = dp.loadAllPatients()
# remove duplicates for ease of classification and building dictionaries later 
seqs[0], seqs[1], vj[0], vj[1], _ =dp.removeDup(seqs[0], seqs[1], vj[0], vj[0])
# now begin the filtration steps. We only want 14 long sequences, 2:10 positions:
seq_mapper=dp.dictEncoder(seqs, vj, filtr=True, clip=True, filtrLen=14)
# the data is now stored in a dictionary and the keys are the X
# need to convert the data to one hot encoded
X_char=list(seq_mapper.keys()) # get the seqs 
# replace AAs with integers
X=dp.char2int(X_char,8)
# get the onehot
X_data = dp.oneHot(X)
X_data = np.array(X_data).reshape(-1,160)

#==============================================================================
# Helper functions
#==============================================================================
def faster(x):
    d = x.reshape(-1, x.shape[-1])
    d2 = np.zeros_like(d)
    d2[np.arange(len(d2)), d.argmax(1)] = 1
    d2 = d2.reshape(x.shape)
    return d2

#==============================================================================
# Build the graph
#==============================================================================

d = 8       # latent dimension
M = 128     # batch size 
maxL= 160   # size of final dimension
lr = 0.001  # learning rate
n_epochs = 10

tf.reset_default_graph()

x_in = tf.placeholder(tf.float32, [None, maxL])
#encoder = slim.fully_connected(x_in, 512, activation_fn=tf.nn.relu)
encoder = slim.fully_connected(x_in, 256, activation_fn=tf.nn.relu)
#encoder = slim.fully_connected(encoder, 128, activation_fn=tf.nn.relu)
#encoder = slim.fully_connected(encoder, 64, activation_fn=tf.nn.relu)
encoder = slim.fully_connected(encoder, 32, activation_fn=tf.nn.relu)
encoded = slim.fully_connected(encoder, d, activation_fn=tf.nn.relu)

decoder = slim.fully_connected(encoded, 32, activation_fn=tf.nn.relu)
#decoder = slim.fully_connected(decoder, 64, activation_fn=tf.nn.relu)
#decoder = slim.fully_connected(decoder, 128, activation_fn=tf.nn.relu)
#decoder = slim.fully_connected(decoder, 256, activation_fn=tf.nn.relu)
decoder = slim.fully_connected(decoder, 160, activation_fn=tf.nn.relu)
decoder = tf.reshape(decoder,(-1,8,20))
decoder = tf.nn.softmax(decoder)
decoded = slim.flatten(decoder)


loss = tf.reduce_mean(tf.square(x_in-decoded))

optimizer = tf.train.AdamOptimizer(learning_rate=lr,
                                    beta1=0.9,
                                    beta2=0.999,
                                    epsilon=1e-08,).minimize(loss)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size=M
n_epoch = 10
for j in range(n_epoch):
    print('----- Epoch', j, '-----')
    
    X = sk.utils.shuffle(X_data,random_state=42)
    n=X.shape[0]            
                
    # Training Loop
    avg_loss = 0.0
    
    for i in range(n // batch_size):
        xT=X[i * batch_size: (i + 1) * batch_size]
        #xT=np.reshape(xT,(-1,self.maxL))
        _,ls = sess.run([optimizer,loss],feed_dict={x_in: xT})
        avg_loss += ls
        
    
    
    avg_loss = avg_loss / n
    #avg_loss = avg_loss / batch_size
    print("MSE: {:0.6f}".format(avg_loss))
    
    parts=X[:2]
    samples = sess.run(decoded,
                       feed_dict={x_in: parts})
    parts = parts.reshape(-1,8,20)
    parts = faster(parts)
    parts = dp.oneHot2AA(parts)
    samples = samples.reshape(-1,8,20)
    samples= faster(samples)
    samples = dp.oneHot2AA(samples)
    
    for idx, samp in enumerate(samples):
        print(samp)
        print()
        print(parts[idx])
        print()
    
y=[]
for val in list(seq_mapper.keys()):
    if seq_mapper[val][0]=='CD4':
        y.append(0)
    else:
        y.append(1)

        
X_data=sess.run(decoded,feed_dict={x_in: X_data})
seqs=None
vj=None

X, y = sk.utils.shuffle(X_data,y)
X_data=None        
xTrain, xVal, yTrain, yVal = train_test_split(X, y, test_size=0.20) 

#neigh=KNN(n_neighbors=5)
#neigh.fit(xTrain, yTrain)
#y_true, y_pred = yVal, neigh.predict(xVal)
#print("{} Validaton Accuracy".format(accuracy_score(y_true, y_pred)))
#print(classification_report(y_true, y_pred))



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















# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 12:21:10 2017

@author: lewismoffat

This script runs a Variational AutoEncoder network on all the data for one-hot
encoded amino acids
"""

#==============================================================================
# IMPORTS
#==============================================================================
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import dataProcessing as dp
import sklearn as sk
import pdb

import edward as ed

from edward.models import Normal, OneHotCategorical
from keras.layers  import Dense
from tensorflow.contrib import slim
#==============================================================================
# Parameters
#==============================================================================
#tf.reset_default_graph()
#==============================================================================
# ORGANIZE THE DATA
#==============================================================================

# load in sequences and vj info for all patients with naive and beta chains
seqs, vj = dp.loadAllPatients()
# remove duplicates for ease of classification and building dictionaries later 
seqs[0], seqs[1], vj[0], vj[1], _ =dp.removeDup(seqs[0], seqs[1], vj[0], vj[0])
# now begin the filtration steps. We only want 14 long sequences, 2:10 positions:
seq_mapper=dp.dictEncoder(seqs, vj, filtr=True, clip=False, filtrLen=14)
# the data is now stored in a dictionary and the keys are the X
# need to convert the data to one hot encoded
X_char=list(seq_mapper.keys()) # get the seqs 
# replace AAs with integers
X=dp.char2int(X_char,8)
# get the onehot
X_data = dp.oneHot(X)
X_data = np.array(X_data).reshape(-1,14,20)


#==============================================================================
# Build the graph
#==============================================================================

d = 8       # latent dimension
M = 128     # batch size 
maxL= 280   # size of final dimension
lr = 0.001  # learning rate
 

def generative_net(z):
    """
    Generative network to parameterize generative model. It takes
    latent variables as input and outputs the likelihood parameters.
    logits = neural_network(z)
    
    """
    net = Dense(128,activation='relu')(z)
    net = Dense(280,activation=None)(net)
    net = tf.reshape(net,(-1,14,20))
    return net
    
def inference_network(x, d):
    """
    Inference network to parameterize variational model. It takes
    data as input and outputs the variational parameters.
    loc, scale = neural_network(x)
    
    """
    net = slim.flatten(x)
    net = Dense(128,activation='relu')(net)
    net = Dense(256, activation='relu')(net)
    
    params = slim.fully_connected(net, d * 2, activation_fn=None)
    
    loc = params[:, :d]
    scale = tf.nn.softplus(params[:, d:])

    return loc, scale
    

# define the generative sub graph model
z = Normal(loc=tf.zeros([M, d]), scale=tf.ones([M, d]))
logits = generative_net(z)
x = OneHotCategorical(logits=logits)
x = tf.cast(x, tf.float32)
# x is now [batch x 14 x 20]


x_ph = tf.placeholder(tf.float32, [M, 14, 20])
loc, scale = inference_network(x_ph,d)
qz = Normal(loc=loc, scale=scale)



inference = ed.KLqp({z: qz}, data={x: x_ph})
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
inference.initialize(optimizer=optimizer)

hidden_rep = tf.sigmoid(logits)

sess = ed.get_session()
init = tf.global_variables_initializer()
init.run()



batch_size=M
n_epoch = 100
for j in range(n_epoch):
    print('----- Epoch', j, '-----')
    
    X = sk.utils.shuffle(X_data,random_state=42)
    n=X.shape[0]            
                
    # Training Loop
    avg_loss = 0.0
    
    for i in range(n // batch_size):
        xT=X[i * batch_size: (i + 1) * batch_size]
        #xT=np.reshape(xT,(-1,self.maxL))
        info_dict = inference.update(feed_dict={x_ph: xT})
        avg_loss += info_dict['loss']


    # Print a lower bound to the average marginal likelihood for a
    # sequence.
    avg_loss = avg_loss / n
    avg_loss = avg_loss / batch_size
    print("log p(x) >= {:0.3f}".format(avg_loss))
    
    # Prior predictive check.
    #seqs = sess.run(z)
    #import pdb; pdb.set_trace()
    #print(seqs[0])

inference.finalize()
    

#
#sess = ed.get_session()
#init = tf.global_variables_initializer()
#init.run()
#
#batch_size=self.batch_size # did this as copied old code
#n=xTrain.shape[0]
#
#
#
#
#











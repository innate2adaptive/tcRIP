# -*- coding: utf-8 -*-
"""
Created on Thu May 25 12:06:33 2017

@author: lewismoffat
"""

"""
Code by Lewis Moffat

"""

import tensorflow as tf
import logging
import numpy as np
import os
import sklearn as sk


class Controller:
    
    def __init__(self, qparams, params):
        """
        Initialize the over all system
        """
        # over network variables
        self.batch_size=qparams['batch_size']
        self.epochs = qparams['epochs']
        
        logging.info('Initialized Learner')
        
        # bootup network
        self.model=Model(params)
        
                                           
    def predict_seq(self, x, phase, s):
        """
        Predicts values of sequence
        """
        loss, result=self.model.pred(x, 0, s) # the zero means its not training
        low_vals= result < 0.5
        result[low_vals]=0
        high_vals=result>=0.5
        result[high_vals]=1
        return result
        
    def train(self, xTrain, yTrain, sqTrain, xVal, yVal, sqVal):
        """
        Runs through the set number of epochs or until its killed
        """
        batch_size=self.batch_size # did this as copied old code
        n=xTrain.shape[0]
        bestScore=0

        for j in range(self.epochs):
            print('----- Epoch', j, '-----')
            
            #Shuffle 
            X, Y, S = sk.utils.shuffle(xTrain, yTrain, sqTrain, random_state=1)
             
            # Training on a small subset of data to make sure it can overfit
#            X=X[:100]
#            Y=Y[:100]
#            S=S[:100]            
            
            n=X.shape[0]            

            # Training Error
            y_pred=self.predict_seq(X,0,S)
            score=sk.metrics.accuracy_score(Y, y_pred)
            print("Training Accuracy: {0:.2f}%".format(score*100))
            
            # Validation Error
            y_pred=self.predict_seq(xVal,0,sqVal)
            score=sk.metrics.accuracy_score(yVal, y_pred)
            print("Validation Accuracy: {0:.2f}%".format(score*100))
            if score>bestScore:
                bestScore=score
            
            # Training Loop
            batchLoss=0
            for i in range(n // batch_size):
                x=X[i * batch_size: (i + 1) * batch_size]
                y=Y[i * batch_size: (i + 1) * batch_size]
                y=np.expand_dims(y,2)
                s=S[i * batch_size: (i + 1) * batch_size]
                batchLoss+= self.model.step(x, y, 1, s)
            print("Loss: {}".format(batchLoss/n))
        
        print("Best Score: {0:.2f}".format(bestScore*100))
            
        return 
        
        
        
class Model:

    def __init__(self, params):
        """
        Initialize the network with a set of parameters in this case params are a dict
        """
        tf.reset_default_graph()
        tf.Graph().as_default()

        
        self.save=params['save']
        self.load=params['load']
        #self.pathshort='../../models/modelA3/'        
        #self.path=self.pathshort+str(params['hidden'])+str(params['learningRate'])+'/model.ckpt'
        
        logging.info('Initialized with following Parameters: {}'.format(params))
        
        self.learningRate=params['learningRate']
        self.regu=params['regu']
        self.maxL=params['maxLen']
        self.cell_size=params['cell_size']
        self.emb_size=params['embedding_size']
        self.vocab_size=params['vocab_size']
        self.lstm=params['LSTM']
        self.stacked=params['stacked']
        self.dropout=params['dropout']
        self.single=params['unidirectional']
        self.attention=params['attention']
        self.attentionLen=params['atten_len']
        self.batch_norm=params['batch_norm']
        self.onlyLinear=params['onlyLinear']
        self.conv=params['conv']
        # stores the session as an attribute
        self.session = self.getPartyStarted() # creates the model
        
                                           
    def getPartyStarted(self):
        """
        This is where the model is defined
        """
        # boot up the placeholders and attach them as attributes
        self.input_holder, self.labels_holder, self.phase_holder, self.seqlen_holder = self.add_place()
        #set random seed
        tf.set_random_seed(80085)
        
        x = self.input_holder
        x = tf.cast(x,tf.float32)
        
        # Graph Code
        
        
        
        # Optimizer Code
        
        
        
        # spool up the session and get the variables initialized
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        
        # both saving and loading need these
        saver = tf.train.Saver()
        path="models/ProtVec/model.ckpt"
    
        # either saves the model or loads it
        if self.load:
            saver.restore(sess,path)   
        else:
            self.sess=sess
            if not os.path.exists(path):
                os.mkdir(path)
            saver = tf.train.Saver()
            saver.save(self.sess, path)
        
        return sess
    
    def add_place(self):
        """
        Adds the place holders to the model
        """
        # set up tf place holders
        x = tf.placeholder(tf.int32, [None, self.maxL])  # input sequences
        y = tf.placeholder(tf.float32, [None,1])           # labels
        phase = tf.placeholder(tf.bool, name='phase')    # true if training
        s = tf.placeholder(tf.int32, [None])             # sequence lengths
        
        return x,y,phase,s 
    
    
    def step(self, x, y, phase, s):
        """
        Defines a single training step on a mini-batch of training samples
        """
        loss, _= self.session.run(
                [self.loss, self.train_optimizer],
                feed_dict = {self.input_holder: x,
                             self.labels_holder: y, 
                             self.phase_holder: phase, 
                             self.seqlen_holder: s  
                             })
        return loss
    
    
    def pred(self, x, phase, s):
        """
        Performs a single prediction based on the currently trainined model
        """     
    
        if str(type(x))=="<class 'scipy.sparse.csr.csr_matrix'>":
            x=x.toarray()
        loss, pred_probs = self.session.run(
                [self.loss, self.preds],
                feed_dict = {self.input_holder: x, 
                             self.labels_holder: np.zeros((x.shape[0],1)),
                             self.phase_holder: phase, 
                             self.seqlen_holder: s
                             })
        
        return loss, pred_probs
        
    def save_model(self):
        path=self.pathshort+str(self.hidden)+str(self.learningRate)+'/'
        if not os.path.exists(path):
            os.mkdir(path)
        saver = tf.train.Saver()
        saver.save(self.session, self.path) 

        




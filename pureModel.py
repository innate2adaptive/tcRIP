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
        
#        # perform nonlinear layer 
#        x = tf.contrib.layers.fully_connected(x, 1024, activation_fn=None, biases_initializer=tf.zeros_initializer) 
#        if self.batch_norm==True:
#            x = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=self.phase_holder) 
#        x = tf.nn.elu(x, 'elu')
        
        # perform nonlinear layer 
        x = tf.contrib.layers.fully_connected(x, 512, activation_fn=None, biases_initializer=tf.zeros_initializer) 
        if self.batch_norm==True:
            x = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=self.phase_holder) 
        x = tf.nn.elu(x, 'elu')
        # perform nonlinear layer 
        x = tf.contrib.layers.fully_connected(x, 128, activation_fn=None, biases_initializer=tf.zeros_initializer) 
        if self.batch_norm==True:
            x = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=self.phase_holder) 
        x = tf.nn.elu(x, 'elu')
        
        # perform the nonlinear layer and softmax layer 
        x = tf.contrib.layers.fully_connected(x, 32, activation_fn=None, biases_initializer=tf.zeros_initializer)
        if self.batch_norm==True:
            x = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=self.phase_holder) 
        x = tf.nn.elu(x, 'elu')
        
        x = tf.contrib.layers.fully_connected(x, 1, activation_fn=None, biases_initializer=tf.zeros_initializer)
        
        self.preds=tf.nn.sigmoid(x)
        
        with tf.name_scope("opt"):
            # sets up learning rate decay
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = self.learningRate
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 10000, 0.99, staircase=True)
             
            #sets ups other optimization details like gradient clipping and L2
            lambda_loss_amount = 0.000001
            Gradient_noise_scale = None
            Clip_gradients = 5.0
            # Loss, optimizer, evaluation
            l2 = lambda_loss_amount * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables() if 'bias' not in tf_var.name)
            #  loss
            if self.regu==True:
                self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=self.labels_holder) + l2)
            else:
                self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=self.labels_holder))
            # Gradient clipping Adam optimizer with gradient noise
            self.train_optimizer = tf.contrib.layers.optimize_loss(
                self.loss,
                global_step=global_step,
                learning_rate=learning_rate,
                optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate),
                clip_gradients=Clip_gradients,
                gradient_noise_scale=Gradient_noise_scale
            )         

#            optimizr=tf.train.AdamOptimizer(learning_rate=self.learningRate)
#            
#            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=self.labels_holder))
#            self.train_optimizer=optimizr.minimize(self.loss)
            
             # spool up the session and get the variables initialized
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())

            if self.load==True:    
                saver = tf.train.Saver()
                saver.restore(sess, self.path)
            
        
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
        loss, pred_probs = self.session.run(
                [self.loss, self.preds],
                feed_dict = {self.input_holder: x, 
                             self.labels_holder: np.zeros((len(x),1)),
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

        




from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import logging
import numpy as np
import os
import sklearn as sk
from edward.models import Normal
from keras.layers import Dense
import edward as ed
from edward.util import Progbar


class Controller:
    
    def __init__(self, qparams, params):
        """
        Initialize the over all system
        """
        # over network variables
        self.batch_size=qparams['batch_size']
        self.epochs = qparams['epochs']
        self.isLoad=params['load']
        self.maxL=params['maxLen']*5
        self.learningRate=params['learningRate']
        
        logging.info('Initialized Learner')
        
        # bootup network
        #self.model=Model(params)
        
                                          
    
        
    def train(self, xTrain, yTrain, sqTrain, xVal, yVal, sqVal):
        """
        Runs through the set number of epochs or until its killed
        """
        
        
        #set random seed
        
        
        d = 10  # latent dimension
        M = self.batch_size  # batch size 
        # Define a subgraph of the full model, corresponding to a minibatch of
        # size M.
        z = Normal(loc=tf.zeros([M, d]), scale=tf.ones([M, d]))
        hidden = Dense(32, activation='relu')(z)
        x = Normal(loc=Dense(self.maxL)(hidden), scale=Dense(self.maxL, activation='softplus')(hidden))
        
        
        
        x_ph = tf.placeholder(tf.float32, [M, self.maxL])
    
        hidden = Dense(32, activation='relu')(x_ph)
        qz = Normal(loc=Dense(d)(hidden), scale=Dense(d, activation='softplus')(hidden))
        
        self.inference = ed.KLqp({z: qz}, data={x: x_ph})
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learningRate)
        self.inference.initialize(optimizer=optimizer)
        
        sess = ed.get_session()
        init = tf.global_variables_initializer()
        init.run()
        
        batch_size=self.batch_size # did this as copied old code
        n=xTrain.shape[0]

        if self.isLoad:
            train=False
        else:
            train=True
            
        if train:
            for j in range(self.epochs):
                print('----- Epoch', j, '-----')
                
                #Shuffle 
                X, Y, S = sk.utils.shuffle(xTrain, yTrain, sqTrain, random_state=1)
                 
                n=X.shape[0]            
                
                # Training Loop
                avg_loss = 0.0
                
                for i in range(n // batch_size):
                    xT=X[i * batch_size: (i + 1) * batch_size]
                    xT=np.reshape(xT,(-1,self.maxL))
                    info_dict = self.inference.update(feed_dict={x_ph: xT})
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
            
            self.inference.finalize()
        #seqs=sess.run(z, {x_ph:X[:128]})
        
        # both saving and loading need these
        saver = tf.train.Saver()
        path="models/varAE/model.ckpt"
        
        # either saves the model or loads it
        if self.isLoad:
            saver.restore(sess,path)   
            import pdb; pdb.set_trace()
        else:
            self.sess=sess
            if not os.path.exists(path):
                os.mkdir(path)
            saver = tf.train.Saver()
            saver.save(self.sess, path)
        
        #Shuffle 
        X, Y, S = sk.utils.shuffle(xTrain, yTrain, sqTrain, random_state=1)
        seqs=sess.run(x, {x_ph:X[:128]})
        print(X[:3])
        print(seqs[:3])
        
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
        self.batch_size=params['batch_size']
        
        # stores the session as an attribute
        self.session = self.getPartyStarted() # creates the model
        
                                           
    def getPartyStarted(self):
        """
        This is where the model is defined
        """
        #with tf.Graph().as_default():
        #set random seed
        tf.set_random_seed(80085)
        ed.set_seed(42)
        
        d = 10  # latent dimension
        M = self.batch_size  # batch size 
        # Define a subgraph of the full model, corresponding to a minibatch of
        # size M.
        z = Normal(loc=tf.zeros([M, d]), scale=tf.ones([M, d]))
        hidden = Dense(32, activation='relu')(z)
        x = Normal(loc=Dense(self.maxL)(hidden), scale=Dense(self.maxL, activation='softplus')(hidden))
        
        
        
        x_ph = tf.placeholder(tf.float32, [M, self.maxL])
    
        hidden = Dense(32, activation='relu')(x_ph)
        qz = Normal(loc=Dense(d)(hidden), scale=Dense(d, activation='softplus')(hidden))
        
        self.inference = ed.KLqp({z: qz}, data={x: x_ph})
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learningRate)
        self.inference.initialize(optimizer=optimizer)
            #init = tf.global_variables_initializer()
            #init.run()

        
        
        
        return #sess
    
    
    
    def step(self, xTrain):
        """
        Defines a single training step on a mini-batch of training samples
        """
        
    
        return info_dict
    
    
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
        path="models/varAE/model.ckpt"
        if not os.path.exists(path):
            os.mkdir(path)
        saver = tf.train.Saver()
        saver.save(self.session, self.path) 

        




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

from keras.layers import Input, Dense, LSTM, RepeatVector
from keras.models import Model
from keras.callbacks import TensorBoard


class Controller:
    
    def __init__(self, qparams, params):
        """
        Initialize the over all system
        """
        # over network variables
        self.batch_size=qparams['batch_size']
        self.epochs = qparams['epochs']
        self.isLoad=params['load']
        self.maxL=params['maxLen']
        self.learningRate=qparams['learningRate']


        logging.info('Initialized Learner')
        
        # bootup network
        self.model=BigModel(params)
        
                                         
        
    def train(self, xTrain, yTrain, xVal, yVal):
        """
        Runs through the set number of epochs or until its killed
        """
#        timesteps = int(self.maxL/5)
#        input_dim = 5
#        
#        xTrain=np.reshape(xTrain,(-1,timesteps,input_dim))
#        xVal=np.reshape(xVal,(-1,timesteps,input_dim))
        
        

        #xTrain = xTrain.toarray()
        #xVal = xVal.toarray()
        
        ## need to use this command line tensorboard --logdir=/tmp/autoencoder
        self.model.autoencoder.fit(xTrain, xTrain,
                epochs=self.epochs,
                batch_size=self.batch_size,
                shuffle=True,
                validation_data=(xVal, xVal),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
        
        
        encoded_seqs=self.model.encoder.predict(xVal)
        decoded_seqs = self.model.decoder.predict(encoded_seqs)
        
        print(xVal[0])
        print(decoded_seqs[0])
        
            
        if tf.gfile.Exists('/tmp/autoencoder'):
            tf.gfile.DeleteRecursively('/tmp/autoencoder') 
        
        xTrain = self.model.encoder.predict(xTrain)
        xVal= self.model.encoder.predict(xVal)
            
        
        return xTrain, xVal
        
        
        
class BigModel:

    def __init__(self, params):
        """
        Initialize the network with a set of parameters in this case params are a dict
        """
        
        tf.Graph().as_default()
        tf.reset_default_graph()
        
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
        self.maxL=params['maxLen']
        
        # stores the session as an attribute
        self.session = self.getPartyStarted() # creates the model
        #self.session = self.lstmVersion()
                                           
    def getPartyStarted(self):
        """
        This is where the model is defined
        """
        
        # this is the size of our encoded representations
        encoding_dim = 8  # 32 floats 
        # this is our input placeholder
        input_seq = Input(shape=(self.maxL,))
        
        if self.maxL>200:
            # "encoded" is the encoded representation of the input
            encoded = Dense(512, activation='sigmoid')(input_seq)
            encoded = Dense(128, activation='sigmoid')(encoded)
            encoded = Dense(32, activation='sigmoid')(encoded)
            encoded = Dense(encoding_dim, activation='sigmoid')(encoded)
            
            # "decoded" is the lossy reconstruction of the input
            decoded = Dense(32, activation='sigmoid')(encoded)
            decoded = Dense(128, activation='sigmoid')(decoded)
            decoded = Dense(512, activation='sigmoid')(decoded)
            decoded = Dense(self.maxL, activation='sigmoid')(decoded)
        else:
            # "encoded" is the encoded representation of the input
            encoded = Dense(64, activation='sigmoid')(input_seq)
            encoded = Dense(32, activation='sigmoid')(encoded)
            encoded = Dense(encoding_dim, activation='sigmoid')(encoded)
            
            # "decoded" is the lossy reconstruction of the input
            decoded = Dense(32, activation='sigmoid')(encoded)
            decoded = Dense(64)(decoded)
            decoded = Dense(self.maxL)(decoded)
        
        
        
        
        # this model maps an input to its reconstruction
        self.autoencoder = Model(input_seq, decoded)
        
        # this model maps an input to its encoded representation
        self.encoder = Model(input_seq, encoded)
        
        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(encoding_dim,))
        
        if self.maxL>200:
            
            # retrieve the last layers of the autoencoder model
            decoder_layer0 = self.autoencoder.layers[-4]
            decoder_layer1 = self.autoencoder.layers[-3]
            decoder_layer2 = self.autoencoder.layers[-2]
            decoder_layer3 = self.autoencoder.layers[-1]
            
            # create the decoder model
            self.decoder = Model(encoded_input, decoder_layer3(decoder_layer2(decoder_layer1(decoder_layer0(encoded_input)))))
            
        else:
        
            # retrieve the last layers of the autoencoder model
            decoder_layer1 = self.autoencoder.layers[-3]
            decoder_layer2 = self.autoencoder.layers[-2]
            decoder_layer3 = self.autoencoder.layers[-1]
            
            # create the decoder model
            self.decoder = Model(encoded_input, decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))
        
        self.autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        
        return
        
        
        
    def lstmVersion(self):
        """
        This is where the model is defined
        """
        
        # this is the size of our encoded representations
        latent_dim = 16  # 16 floats 
        timesteps = int(self.maxL/5)
        input_dim = 5
        
        
        inputs = Input(shape=(timesteps, input_dim))
        encoded = LSTM(latent_dim)(inputs)
        
        decoded = RepeatVector(timesteps)(encoded)
        decoded = LSTM(input_dim, return_sequences=True)(decoded)
        
        self.autoencoder = Model(inputs, decoded)
        self.encoder = Model(inputs, encoded)
        
        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(timesteps, latent_dim))
        # retrieve the last layer of the autoencoder model
        decoder_layer = self.autoencoder.layers[-1]
        # create the decoder model
        self.decoder = Model(encoded_input, decoder_layer(encoded_input))
        
        self.autoencoder.compile(optimizer='adam', loss='mean_squared_error')
       
        return
    
    

        




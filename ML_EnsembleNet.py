#==============================================================================
# IMPORTS
#==============================================================================
import tensorflow as tf
import numpy as np
import os
import dataProcessing as dp
import sklearn as sk

from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

import keras
from keras.models import Model
from keras.layers import Input, Dense



#==============================================================================
# ORGANIZE THE DATA
#==============================================================================


mb_size = 32
Z_dim = 16

# load in sequences and vj info for all patients with naive and beta chains
seqs, vj = dp.loadAllPatients()

# filter out joint sequences
seqs[0], seqs[1], vj[0], vj[1], joint = dp.removeDup(seqs[0], seqs[1], vj[0], vj[1])

print("Number of Shared Seqs: {}".format(len(joint)))
print("Shared Percent: %.2f%%" % (len(joint)/(len(seqs[0])+len(seqs[1])) * 100.0))

# add extrac cds
cddict=dp.extraCDs()
seqs[0], vj[0]=dp.addCDextra(seqs[0],vj[0],cddict)
seqs[1], vj[1]=dp.addCDextra(seqs[1],vj[1],cddict)


# replace the seqeunces with their atchely vectors and add the v region as the
# final region
for idx, seqz in enumerate(seqs):
    seqs[idx]=dp.seq2fatch(seqz) 
    for idx2, seq in enumerate(seqs[idx]):
        new=np.zeros((1))
        new[0]=float(vj[idx][idx2][0])
        seqs[idx][idx2]=np.concatenate((seq,new))
    

# filter to a set length
length=14+5+6 # 14 is the most abundant
seqs[0]=np.array(dp.filtr(seqs[0], length*5+1))
seqs[1]=np.array(dp.filtr(seqs[1], length*5+1))

# use function to create data
X, y = dp.dataCreator(seqs[0],seqs[1])

# shuffle data
X, y = sk.utils.shuffle(X,y)


# print class balances
dp.printClassBalance(y)


# because of keras I need to expand the y so that its one-hot encoded
def quickExpand(y):
    newY=[]
    for val in y:
        small=[0,0]
        small[int(val)]+=1
        newY.append(small)
    newY=np.array(newY)
    return newY
    
y=quickExpand(y)


# 25% Validation set
xTrain, xVal, yTrain, yVal= train_test_split(X, y, test_size=0.25) 

print("======================================")
print("Running Classification using Keras NN")


# keras is lacking in that it cant handle splitting

cdr1t, cdr2t, cdr3t, v_genet, _ = np.split(xTrain,(25,55,125,126),1)
cdr1v, cdr2v, cdr3v, v_genev, _ = np.split(xVal,(25,55,125,126),1)


# defining the keras model
cdr1in = Input(shape=(25,))
cdr2in = Input(shape=(30,))
cdr3in = Input(shape=(70,))
v_genein = Input(shape=(1,)) 

#CDR1 pipeline
cdr1 = Dense(64, activation='elu')(cdr1in)
cdr1_2 = Dense(64, activation='elu')(cdr1)
cdr1_3 = Dense(2, activation='softmax')(cdr1_2)

#CDR2 pipeline
cdr2 = Dense(64, activation='elu')(cdr2in)
cdr2_2 = Dense(64, activation='elu')(cdr2)
cdr2_3 = Dense(2, activation='softmax')(cdr2_2)

#CDR3 pipeline
cdr3 = Dense(128, activation='elu')(cdr3in)
cdr3_2 = Dense(64, activation='elu')(cdr3)
cdr3_3 = Dense(2, activation='softmax')(cdr3_2)

#CDR1 pipeline
v_gene = Dense(64, activation='elu')(v_genein)
v_gene_2 = Dense(64, activation='elu')(v_gene)
v_gene_3 = Dense(2, activation='softmax')(v_gene_2)

# bring it back together
#x = tf.concat((cdr1_3, cdr2_3, cdr3_3, v_gene_3),1)

x_1 = keras.layers.concatenate([cdr1_3, cdr2_3, cdr3_3, v_gene_3],1)
    
x = Dense(64, activation='relu')(x_1)
predictions = Dense(2, activation='softmax')(x)


# define the model
model = Model(inputs=[cdr1in, cdr2in, cdr3in, v_genein], outputs=[predictions])


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x=[cdr1t, cdr2t, cdr3t, v_genet], 
          y=yTrain, 
          batch_size=32, 
          epochs=10, 
          verbose=2,  
          validation_split=0.1)

# evaluate the model
scores = model.evaluate([cdr1v, cdr2v, cdr3v, v_genev], yVal)
print("\nTesting Accuracy: %.2f%%" % ( scores[1]*100))



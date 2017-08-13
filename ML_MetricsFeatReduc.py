# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 11:15:21 2017

@author: lewismoffat

This scripts runs the metrics based feature reduction on several amino acid feature engineering methods

"""

#==============================================================================
#  Imports
#==============================================================================
import numpy              as np
import dataProcessing     as dp
import sklearn            as sk
import matplotlib.pyplot  as plt
import pdb


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif


#==============================================================================
#  Parameters
#==============================================================================
loadCDR12=True     #include cdr1 and cdr2
addV=False          #include the v region
atchley=True        #use the atchley vectors for feat. eng.
length=14           #length to filter the seqeunces to 

#==============================================================================
# Pull in sequences
#==============================================================================
# load in sequences and vj info for all patients with naive and beta chains
seqs, vj = dp.loadAllPatients()
# remove duplicates for ease of classification and building dictionaries later 
seqs[0], seqs[1], vj[0], vj[1], _ =dp.removeDup(seqs[0], seqs[1], vj[0], vj[0])

if loadCDR12:
    # add extrac cds
    cddict=dp.extraCDs()
    seqs[0], vj[0]=dp.addCDextra(seqs[0],vj[0],cddict)
    seqs[1], vj[1]=dp.addCDextra(seqs[1],vj[1],cddict)
   
#==============================================================================
#  Feature Engineering
#==============================================================================
if atchley:  
    print("Using Atchley Feat. Eng.")
    if loadCDR12:
        print("Adding CDR1 and CDR2")
        # replace the seqeunces with their atchely vectors and add the v region as the
        # final region
        for idx, seqz in enumerate(seqs):
            seqs[idx]=dp.seq2fatch(seqz) 
            if addV:
                for idx2, seq in enumerate(seqs[idx]):
                    new=np.zeros((1))
                    new[0]=float(vj[idx][idx2][0])
                    seqs[idx][idx2]=np.concatenate((seq,new))
                
        print("Filtering to {} Long".format(length))
        # filter to a set length
        length=14+5+6 # 14 is the most abundant
        if addV:
            length+=1
        seqs[0]=np.array(dp.filtr(seqs[0], length*5))
        seqs[1]=np.array(dp.filtr(seqs[1], length*5))
        
    else:
        # replace the seqeunces with their atchely vectors and add maybe the v
        #region as the final region
        print("Filtering to {} Long".format(length))
        seqs[0]=dp.filtr(seqs[0], length)
        seqs[1]=dp.filtr(seqs[1], length)
        
        for idx, seqz in enumerate(seqs):
            seqs[idx]=dp.seq2fatch(seqz) # note that this expects a list 
            if addV:
                for idx2, seq in enumerate(seqs[idx]):
                    seqs[idx][idx2]=np.concatenate((seq,vj[idx][idx2]))
                
        
        # filter to a set length
        # 14 is the most abundant

# use function to create data
X, y = dp.dataCreator(seqs[0],seqs[1])

# comment this in to reduce the number of training points in the set to improve 
# the time for feature reduction.
X=X[:10000]
X=X[:,70:]
y=y[:10000]

# shuffle data
X, y = sk.utils.shuffle(X,y)

print("selecting K_best features")
# boot up the class to select the k best features
sel=SelectKBest(mutual_info_classif, k=10)
# transform X with the k best features
sel.fit_transform(X, y)    
scores=np.expand_dims(sel.scores_,1)
scores=np.reshape(scores,(-1,5))
scores=np.mean(scores,axis=1)
scores=np.expand_dims(scores,1).T
plt.imshow(scores,cmap='plasma_r')
plt.xlabel("AA Position")
#plt.xticks(np.arange(0,14),["1","2","3","4","5","6","7","8","9","10","11","12","13","14"])
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left='off',      # ticks along the bottom edge are off
    right='off',         # ticks along the top edge are off
    labelleft='off') # labels along the bottom edge are off
plt.title("Mutual Information Importance of Amino Acid Position")
plt.show()

print(sel.scores_)

    
    


# -*- coding: utf-8 -*-
"""
Created on Thu May 25 11:15:56 2017

@author: lewismoffat
"""
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
import atchFactors as af
from sklearn.cluster import MiniBatchKMeans


# putting the dictionary globally is naughty but its called so often its worth it
intDict={ 'A': 1 ,
          'C': 2 ,
          'D': 3 ,
          'E': 4 ,
          'F': 5 ,
          'G': 6 ,
          'H': 7 ,
          'I': 8 ,
          'K': 9 ,
          'L': 10 ,
          'M': 11 ,
          'N': 12 ,
          'P': 13 ,
          'Q': 14 ,
          'R': 15 ,
          'S': 16 ,
          'T': 17 ,
          'V': 18 ,
          'W': 19 ,
          'Y': 20 ,
          'X': 21 ,
          'Z': 21 ,
          'U': 21 ,
          '0': 0 }



def char2int(seqs, longest):
    """converts all characters in a sequnce to integer IDs based on a dict"""
    # iterate through sequences
    for index, seq in enumerate(seqs):
        # pad sequence with zeros to get to the longest length
        seq=seq.ljust(longest,'0')
        # temporary var for storing int seq before replacement
        numseq=[]
        # go character by character and fill new list with numeric values
        for char in seq:
            numseq.append(intDict[char])
        seqs[index]=numseq
    # convert to a numpy array for convenience later
    seqs=np.array(seqs)
    return seqs
    
def char2ptuple(seqs, n=3):
    
    # encodes feature dictionaries as numpy vectors, needed by scikit-learn.
    vectorizer = DictVectorizer(sparse=True)
    newSeqs = vectorizer.fit_transform([event_feat(x, n) for x in seqs])
    return newSeqs


def clip(seqs, ln):
    """Goes through list of sequences and clips the sequence to ln characters long"""
    for idx, seq in enumerate(seqs):
        seqs[idx]=seq[:len]
    return seqs

def filtr(seqs, ln):
    """Goes through list of sequences and clips the sequence to ln characters long"""
    newSeq=[]
    for idx, seq in enumerate(seqs):
        if len(seq)==ln:
            newSeq.append(seq)
    return newSeq
    
def seq2fatch(seqs):
    for idx, seq in enumerate(seqs):
        vec=[]
        for char in seq:
            vec=np.concatenate((vec,af.atchleyFactor(char)))
        seqs[idx]=vec
    seqs=np.array(seqs)
    return seqs
    
#==============================================================================
# Unsupervized clustering of pTuples using k-means    
#==============================================================================

def kmeans(seqs,n=3,sample=10000, num_clusters=100):
    # first step is to get ptuples and replace them with atchley vectors
    # the idea is a ptuple 1x15 from here is a data point for kmeans so we just want one big list
    # first we want to go through each seq in the sequence
    #n=3 # size of the tuple 
    newSeq=[] # temp vector to fill atchely numbers in
    for idx, seq in enumerate(seqs):
        for i in range(len(seq)-n+1):
            tup=seq[i:i+n]
            tup=np.concatenate((af.atchleyFactor(tup[0]),af.atchleyFactor(tup[1]),af.atchleyFactor(tup[2])))
            newSeq.append(tup)
            
    newSeq=np.array(newSeq)
    
    
    # for efficiency we will sample from the data and run on that (essentially boostrapping)
    np.random.shuffle(newSeq)
    newSeq=newSeq[:sample]
    
    # fit kmeans 
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, verbose=0).fit(newSeq)
    
    freqVec=np.zeros(num_clusters)
    newSeqs=[]
    # can now use this to predict points - replace 
    for idx, seq in enumerate(seqs):
        tuples=atch_pTuple(seq,n)
        preds=kmeans.predict(tuples)
        for val in preds:
            freqVec[int(val)]+=1
        newSeqs.append(freqVec)
        freqVec=np.zeros(num_clusters)
    newSeqs=np.array(newSeqs)
    
    return newSeqs
    
#==============================================================================
# # Helper Functions for constructing pTuples
#==============================================================================
def atch_pTuple(seq,n=3):
    point=[]
    for i in range(len(seq)-n+1):
        tup=seq[i:i+n]
        tup=np.concatenate((af.atchleyFactor(tup[0]),af.atchleyFactor(tup[1]),af.atchleyFactor(tup[2])))
        point.append(tup)
    point=np.array(point)
    return point
    
def pTuple(vec,n=3):
    """Returns a vector of ptuples from a given sequence"""
    return [vec[i:i+n] for i in range(len(vec)-n+1)]

def event_feat(event, n=3):
    ####### Creates Dictionary ########
    result = defaultdict(float)
    event=pTuple(event, n)
    for tup in event:      
        if "X" in tup or "Z" in tup or "U" in tup or "B" in tup :
            continue
        result[tup]+=1
    return result

                               
    
    
    
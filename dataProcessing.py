# -*- coding: utf-8 -*-
"""
Created on Thu May 25 11:15:56 2017

@author: lewismoffat
"""
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer

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



    
def char2ptuple(seqs):
    
    # encodes feature dictionaries as numpy vectors, needed by scikit-learn.
    vectorizer = DictVectorizer(sparse=False)
    newSeqs = vectorizer.fit_transform([event_feat(x) for x in seqs])
    return newSeqs


def clip(seqs, ln):
    """Goes through list of sequences and clips the sequence to ln characters long"""
    for idx, seq in enumerate(seqs):
        seqs[idx]=seq[:len]
    return seqs

def filter(seqs, ln):
    """Goes through list of sequences and clips the sequence to ln characters long"""
    newSeq=[]
    for idx, seq in enumerate(seqs):
        if len(seq)==14:
            newSeq.append(seq)
    return newSeq
    
# Helper Functions for constructing pTuples

def pTuple(vec,n=3):
    """Returns a vector of ptuples from a given sequence"""
    return [vec[i:i+n] for i in range(len(vec)-n+1)]

def event_feat(event):
    ####### Creates Dictionary ########
    result = defaultdict(float)
    event=pTuple(event)
    for tup in event:      
        if "X" in tup or "Z" in tup or "U" in tup or "B" in tup :
            continue
        result[tup]+=1
    return result

                               
    
    
    
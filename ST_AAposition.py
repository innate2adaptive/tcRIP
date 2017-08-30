# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 17:05:48 2017

@author: lewismoffat

This generates heat maps of amino acids used in different positions of a 
sequence of set length e.g. 14 AA long. It also generates some basic stats 
of the most common V-J combo.

"""

#==============================================================================
# Module Imports
#==============================================================================
import numpy as np
import dataProcessing as dp
import pdb
import matplotlib.pyplot as plt
from collections import Counter


from scipy.stats import ks_2samp
from scipy.stats import chisquare

#==============================================================================
# Load data
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

# filter to 14 long, together its still 200k seqs
seqs[0]=dp.filtr(seqs[0], 14)
seqs[1]=dp.filtr(seqs[1], 14)

#==============================================================================
# Analysis of V-J region being used
#==============================================================================
sepVJ=False
if sepVJ:
    # get the vjs amd convert them to strings which are put in a counter
    vjs=[str(x[1][0])+','+str(x[1][1]) for x in list(seq_mapper.values())]
    # The most common cluster with 3874 examples is 15,12
    vjc=Counter(vjs)
    newCD4s=[]
    newCD8s=[]
    for key in list(seq_mapper.keys()):
        row=seq_mapper[key]
        if row[1]==[15,12]: # the most common combo
            if row[0]=="CD4":
                newCD4s.append(key)
            else:
                newCD8s.append(key)





#==============================================================================
# Calculate the Postional Prob
#==============================================================================

# This is used quite often so laying it here
intDictzero={ 'A': 0 ,'C': 0 ,'D': 0 ,'E': 0 ,'F': 0 ,'G': 0 ,'H': 0 ,'I': 0 ,'K': 0 ,'L': 0 ,'M': 0 ,
              'N': 0 ,'P': 0 ,'Q': 0 ,'R': 0 ,'S': 0 ,'T': 0 ,'V': 0 ,'W': 0 ,'Y': 0 ,}

def positionProb(seqs, length, intDictzero):
    # this function takes a list of sequences and for each position it calculates
    # a distribution
    
    # we only want to look at each sequence once so we want to have the e.g. 14
    # positional dictionaries already made
    
    # this stores the dictionaries
    listDict=[]
    for i in range(length):
        # for every position add a dictionary copy of intDictzero
        listDict.append(intDictzero.copy())
    # iterate through every sequence
    for seq in seqs:
        # iterate through every amino acid 
        for idx, aa in enumerate(seq):
            # use its position to access the dict and add one for the amino acid present
            listDict[idx][aa]+=1
    
    dictLookup=[]
    # now take the values and create a numpy array while normalizing 
    for idx, dictionary in enumerate(listDict):
        # this normalizes the values
        factor=1.0/sum(dictionary.values())
        for k in dictionary:
          dictionary[k] = dictionary[k]*factor
        
        dictLookup.append(dictionary)
        
        newVals=[]
        for char in sorted(dictionary):# this enforces alphabetical ordering
            newVals.append(dictionary[char])
        listDict[idx]=np.array(newVals)
        # turns dict values into a numpy array and then replaces the original
        #listDict[idx]=np.fromiter(iter(dictionary.values()), dtype=float)
        #pdb.set_trace()
    listDict=np.array(listDict)
        
    return listDict, dictLookup
    

# its handy to have the look up dictionaries for 
Prob4, dl4=positionProb(seqs[0], 14, intDictzero)
Prob8, dl8=positionProb(seqs[1], 14, intDictzero)



plt.imshow(Prob4.T[:,4:10], cmap='plasma')
plt.title('CD4 Heat Map of AA usage for clipped 14 long AAs')
plt.xlabel('AA Position')
plt.ylabel('Amino Acid')
plt.xticks(np.arange(6),['5','6','7','8','9','10'])
plt.yticks(np.arange(20),['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'])
plt.show()

plt.imshow(Prob8.T[:,4:10], cmap='plasma')
plt.title('CD8 Heat Map of AA usage for clipped 14 long AAs')
plt.xlabel('AA Position')
plt.ylabel('Amino Acid')
plt.xticks(np.arange(6),['5','6','7','8','9','10'])
plt.yticks(np.arange(20),['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'])
plt.show()

plt.imshow(Prob4.T[:]-Prob8.T[:], cmap='plasma')
plt.title('Difference in AA usage for 14 long AAs')
plt.xlabel('AA Position')
plt.ylabel('Amino Acid')
plt.xticks(np.arange(14))
plt.yticks(np.arange(20),['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'])
plt.show()



# funtion that replaces each sequence with its probability position values
def charForProb(seqs,dic):
    newSeqs=[] # will contain all new sequences
    for seq in seqs:
        run=[] # will contain the new prob vals for a sequence
        for idx, char in enumerate(seq):
            run.append(dic[idx][char]) # go through and replace char with prob
        #run=np.log(np.product(run)+1) # can add if you want it to be stable
        newSeqs.append(run)
    return newSeqs







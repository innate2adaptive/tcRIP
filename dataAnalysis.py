# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 11:50:22 2017

@author: lewismoffat

This script first uses the standard method for data extraction to then analyze
the ptuples and amino acid usage in different sequences.
"""

########################################################
# Imports
########################################################

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import dataProcessing as dp
import pdb
from collections import defaultdict
########################################################
# Boolean switches for running
########################################################

onlyCD3     = True # this means files contain CDR1/2 sequences so have to be parsed differently


########################################################
# File Names
########################################################

# File names for different data; A - alpha chain, B - beta chain            
cd4A_file = 'patient1/vDCRe_alpha_EG10_CD4_naive_alpha.txt'
cd8A_file = 'patient1/vDCRe_alpha_EG10_CD8_naive_alpha.txt'
cd4B_file = 'patient1/vDCRe_beta_EG10_CD4_naive_beta.txt'
cd8B_file = 'patient1/vDCRe_beta_EG10_CD8_naive_beta.txt'
data = 'data/'
extra = 'extra/'


########################################################
# Data Retrieval 
########################################################
"""
The data for the first patient is stored in the 'data/patient1' file
where each sequence is encoded as a string and comma separated with its count
at current extraction is unique and count is ignored
"""
# Files to be read
files = [cd4B_file, cd8B_file]

# sequence list to be filled. Make sure file order is the same as a below
cd4=[]
cd8=[]
seqs=[cd4,cd8]

# if the data files you want only have cdr3s it has to parse it differently

if onlyCD3:
    for index, file in enumerate(files):
        file=data+file
        with open(file,'r') as infile:
            # goes through each of the files specified in read mode and pulls out 
            # each line and formats it so a list gets X copies of the sequence 
            for line in infile:
                twoVals=line.split(", ")
                twoVals[1]=twoVals[1].replace("\n","")
                for i in range(int(twoVals[1])):
                    seqs[index].append(twoVals[0])
else:
    for index, file in enumerate(files):
        file=data+extra+file
        with open(file,'r') as infile:
            # goes through each of the files specified in read mode and pulls out 
            # each line adds each sequence from the line 
            for line in infile:
                threeVals=line.split(",")
                threeVals[2]=threeVals[2].replace("\n","")
                for i in threeVals:
                    if i=="":
                        continue
                    seqs[index].append(i)


print("CD4 to CD8 Ratio {}:{}".format(len(cd4)/(len(cd4)+len(cd8)),len(cd8)/(len(cd4)+len(cd8))))
print("Total Sequences: {}".format(len(cd4)+len(cd8)))

    


########################################################
# Length Histogram 
########################################################

# at this point both cd4 and cd8 are filled with lists of sequences as strings
# counts have been factored in

# This short code block prints a graph with a bar chart histogram of the 
# length distribution for both CD4 and CD8 sequences together.
# Flip the boolean to enable/disable it


# length histogram
lenArr=[]
# create new vector of lengths 
for seq in cd4+cd8:
    lenArr.append(len(seq))
    
# convert list to numpy array
lenArr=np.asarray(lenArr)

# bins
binNum=max(lenArr)-min(lenArr)
bins=np.arange(min(lenArr),max(lenArr))

# setup graph
plt.hist(lenArr, bins=binNum)
plt.xticks(bins)
plt.title("Sequence Length Histogram")
plt.xlabel("Sequence Length (AA)")
plt.ylabel("Frequency")
plt.show()


########################################################
# Separate to Five Most Common
########################################################


lengthCount=Counter(lenArr)
fiveMostCommon=np.array(lengthCount.most_common(n=5))
print(fiveMostCommon)
fiveMostCommon=fiveMostCommon[:,0]

shrtCd4=[]
shrtCd8=[]
both =[shrtCd4, shrtCd8]
for seq in cd4:
    if len(seq) in fiveMostCommon:
        shrtCd4.append(seq)
    
for seq in cd8:
    if len(seq) in fiveMostCommon:
        shrtCd8.append(seq)        

        
        
cd4=shrtCd4.copy()
cd8=shrtCd8.copy()
        
print("\nFiltered to five most common lengths")
print("CD4 to CD8 Ratio {}:{}".format(len(shrtCd4)/(len(shrtCd4)+len(shrtCd8)),len(shrtCd8)/(len(shrtCd4)+len(shrtCd8))))
print("Total Sequences: {}".format(len(shrtCd4)+len(shrtCd8)))

########################################################
# Separate and count pTuples
########################################################

for seqs in both:
    for idx, seq in enumerate(seqs):
        ptup=dp.pTuple(seq,n=1)
        seqs[idx]=ptup

# flattens list
shrtCd4s = [item for sublist in shrtCd4 for item in sublist]    
# flattens list
shrtCd8s = [item for sublist in shrtCd8 for item in sublist]        


print("Number of Tuples: {}".format(len(shrtCd4)+len(shrtCd8)))

cd4Count=Counter(shrtCd4s)
cd8Count=Counter(shrtCd8s)

fiveMostCommon1=np.array(cd4Count.most_common(n=20))
print(fiveMostCommon1)  

fiveMostCommon2=np.array(cd8Count.most_common(n=20))
print(fiveMostCommon2)

fiveMostCommon1=fiveMostCommon1[:,1]
fiveMostCommon2=fiveMostCommon2[:,1]

fiveMostCommon1=list(map(int, fiveMostCommon1))
fiveMostCommon2=list(map(int, fiveMostCommon2))
fiveMostCommon1=fiveMostCommon1/np.sum(fiveMostCommon1)
fiveMostCommon2=fiveMostCommon2/np.sum(fiveMostCommon2)



area=[]
totalLen=[]
for seq in shrtCd8:
   for idx, tup in enumerate(seq):
       if tup=='ASSL':
           area.append(idx)  
           totalLen.append(len(seq))    
           


# convert list to numpy array
lenArr=np.asarray(area) 

# bins
binNum=max(lenArr)-min(lenArr)
bins=np.arange(min(lenArr),max(lenArr))

# setup graph
plt.hist(lenArr, bins=binNum)
plt.xticks(bins)
plt.title("Sequence Length Histogram")
plt.xlabel("Sequence Length (AA)")
plt.ylabel("Frequency")
plt.show()

# convert list to numpy array
lenArr=np.asarray(totalLen)

# bins
binNum=max(lenArr)-min(lenArr)
bins=np.arange(min(lenArr),max(lenArr))

# setup graph
plt.hist(lenArr, bins=binNum)
plt.xticks(bins)
plt.title("Sequence Length Histogram")
plt.xlabel("Sequence Length (AA)")
plt.ylabel("Frequency")
plt.show()

plt.scatter(area, totalLen)
plt.show()

########################################################
# Look at individual position distributions
########################################################

# first distribution is at an given location [0,1} in a sequence 
## location is position/length e.g. 3/11
# The frequency of particular amino acid at that location. Can ve visualized in a 3d graph

# as a test just looking at the first position
# putting the dictionary globally is naughty but its called so often its worth it
intDictzero={ 'A': 0 ,
              'C': 0 ,
              'D': 0 ,
              'E': 0 ,
              'F': 0 ,
              'G': 0 ,
              'H': 0 ,
              'I': 0 ,
              'K': 0 ,
              'L': 0 ,
              'M': 0 ,
              'N': 0 ,
              'P': 0 ,
              'Q': 0 ,
              'R': 0 ,
              'S': 0 ,
              'T': 0 ,
              'V': 0 ,
              'W': 0 ,
              'Y': 0 ,
              '0': 0 }
              
intDictIndex={ 'A': 1 ,
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
              '0': 0 }
          
              
              
aas=['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
surface=[]

for aa in aas:
    surface=[]

    for seq in cd8:
        for idx, char in enumerate(seq):
            if char==aa:
                seqLen=len(seq)
                surface.append((idx+1)/seqLen)
    surface=np.array(surface) 
    # setup graph
    counter=Counter(surface)
    bins=len(counter.keys())
    plt.hist(surface, bins=bins)
    plt.title("Amino Acid Usage Histogram: {}".format(aa))
    plt.xlabel('Normalized position in Sequence')
    plt.show()

#for seq in cd4:
#    seqLen=len(seq)
#    for idx, char in enumerate(seq):
#        surface.append([(idx+1)/seqLen,intDictIndex[char]])
  

# setup graph
#plt.scatter(surface[:,0],surface[:,1])
#plt.yticks(aas)

# setup graph
#counter=Counter(surface)
#bins=len(counter.keys())
#plt.hist(surface, bins=bins)
#plt.title("Amino Acid Usage Histogram: {}".format(aa))
#plt.xlabel('Normalized position in Sequence')
#plt.show()









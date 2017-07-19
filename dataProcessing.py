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
import pandas as pd
import sklearn as sk
from tqdm import tqdm

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
          'X': 0 ,
          'Z': 0 ,
          'U': 0 ,
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
 
    
def GloVe(seqs, swissprot=True):
    
    if swissprot==True:
        df=pd.read_csv('embeddings/protVec_100d_3grams.csv', sep="\t" ,header=None)
        notformatted=[]
        for row in df.values:
            
            row=row[0].split("\t")
            key=row[0]
            vals=np.array([float(x) for x in row[1:]])
            notformatted.append([key, vals])
            
        dictionary=dict(notformatted)
    else:
        dictionary=np.load('embeddings/dict_norm.npy')
        dictionary=dict(dictionary.item())
    
    
    
    
    
    if swissprot==True:
        for idx, seq in enumerate(seqs):
            newSeq=np.zeros((dictionary['AAA'].shape[0]))
            tuples=pTuple(seq)
            for tup in tuples:
                try:
                    location=dictionary[tup]
                except:
                    location=dictionary["<unk>"]
                newSeq+=location
            seqs[idx]=newSeq
    else:
        embeddings=np.load('embeddings/embed.npy')
        
        for idx, seq in enumerate(seqs):
            newSeq=np.zeros((embeddings.shape[1]))
            tuples=pTuple(seq,4)
            for tup in tuples:
                try:
                    location=dictionary[tup]
                    
                except:
                    try:    
                        location=dictionary["UNK"]
                    except:
                        import pdb; pdb.set_trace()
                newSeq+=embeddings[location]
            seqs[idx]=newSeq
            
    return seqs



def expandTuples(seqs,n=4):
    seqsNew=[]
    for idx, seq in enumerate(seqs):
        tuples=pTuple(seq,n)
        for tup in tuples:
            seqsNew.append(tup)
    return seqsNew    
    
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
            if n==3:
                tup=np.concatenate((af.atchleyFactor(tup[0]),af.atchleyFactor(tup[1]),af.atchleyFactor(tup[2])))
            else:
                tup=np.concatenate((af.atchleyFactor(tup[0]),af.atchleyFactor(tup[1])))
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
        if n==3:
            tup=np.concatenate((af.atchleyFactor(tup[0]),af.atchleyFactor(tup[1]),af.atchleyFactor(tup[2])))
        else:
            tup=np.concatenate((af.atchleyFactor(tup[0]),af.atchleyFactor(tup[1])))
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

#==============================================================================
# # Helper Functions DataSet Creation
#==============================================================================                              
def dataReader(files, delim):
    """ 
    This takes a list of file names and a list deliminators which are strings
    and reads the data from them. Expects decombinator file names.
    THIS IGNORES THE NUCLEOTIDE ADDITION SEQUENCE
    """
    cd4=[]
    cd8=[]
    if not files:
        print("please provide a list of file names")
        return
    else:
        # go through each file in the liss
        for file in files:
                # using an imap to check all delimitors exist
                if all(map(file.__contains__, delim)):
                    with open(file,'r') as infile:
                        # goes through each of the files specified in read mode and pulls out 
                        # each line and formats it so a list gets X copies of the sequence 
                        for line in infile:
                            line=lineCleaner(line)
                            if "CD4" in file:
                                cd4.append(line)
                            elif "CD8" in file:
                                cd8.append(line)
                            
    return cd4, cd8
        
def lineCleaner(string):
    """
    Cleaning a line supplied like below and getting a list of important shit
    e.g. '20, 6, 4, 0, CTAGGAG, 1, CAWSPKGPQETQYF, 1\n'
    """
    string=string.replace("\n","")
    string=string.split(", ")
    string=string[0:4]+[string[6]]
    return string

def dataSpliter(cd, combine=True):
    """ 
    This gets the v region and j region and sequence region
    """
    if combine:
        vj=[]
        seqs=[]
        for line in cd:
            vj.append([line[0],line[0]])
            seqs.append(line[4])
        return  seqs, vj
    else:
        v=[]
        j=[]
        seqs=[]
        for line in cd:
            v.append(line[0])
            j.append(line[1])
            seqs.append(line[4])
        return  seqs, v, j
    return

def dataCreator(cd4,cd8):
    # from this point it is assumed cd4/8 are NUMPY vectors
    # labels are created and then the X and Y vectors are shuffled and combo'd
    # extra contains a list of anything else to be shuffled
    y4=np.zeros((len(cd4)))
    y8=np.zeros((len(cd8)))
    y4[:]=0
    y8[:]=1
    
    # combine classes
    Y = np.concatenate((y4,y8),0)
    X = np.concatenate((cd4,cd8),0)
    
    
    
    return X, Y
    
def printClassBalance(y):
    # this assumes y contains binary classes of 1 and 0
    y1=0
    y0=0
    for yi in y:
        if yi==1:
            y1+=1
        else:
            y0+=1
    print("Class Balance {}:{}".format(y1/(y1+y0),y0/(y1+y0)))
    print("Class Sizes   {}:{}".format(y1,y0))
    
def removeDup(cd4, cd8, v4, v8):
    """ 
    This takes two lists of strings and removes the common sequences between 
    the two, and returns them as a separate string for analytics
    """
    print("Removing Shared Sequences")
    # set up dictionaries so we have the mapping of seq to v
    dicCD4={}
    dicCD8={}
    for idx, seq in enumerate(cd4):
        dicCD4[seq]=v4[idx]
    for idx, seq in enumerate(cd8):
        dicCD8[seq]=v8[idx]
    
    
    joint=list(frozenset(cd4).intersection(cd8))    
    newCD4=list(frozenset(cd4).difference(cd8)) 
    newCD8=list(frozenset(cd8).difference(cd4)) 
    
    v4new=[dicCD4[seq] for seq in newCD4]    
    v8new=[dicCD8[seq] for seq in newCD8]    


    return newCD4, newCD8, v4new, v8new, joint
    
    
#==============================================================================
# Data Loaders    
#==============================================================================

"""
Hard coding the links to the data is sloppy but its easy
"""


def extraCDs():
    """
    This builds two dictionaries (one for each pop.) that maps a cdr3 to its 
    cdr1 and cdr2
    """
    # File names for different data; A - alpha chain, B - beta chain            
    cd4A_file = 'patient1/vDCRe_alpha_EG10_CD4_naive_alpha.txt'
    cd8A_file = 'patient1/vDCRe_alpha_EG10_CD8_naive_alpha.txt'
    cd4B_file = 'patient1/vDCRe_beta_EG10_CD4_naive_beta.txt'
    cd8B_file = 'patient1/vDCRe_beta_EG10_CD8_naive_beta.txt'
    data = 'data/'
    extra = 'extra/'
    # Files to be read
    files = [cd4B_file, cd8B_file]
    
    # sequence list to be filled. Make sure file order is the same as a below
    cd4=[]
    cd8=[]
    seqs=[cd4,cd8]   # this contains the sequences 

    for index, file in enumerate(files):
            file=data+extra+file
            with open(file,'r') as infile:
                # goes through each of the files specified in read mode and pulls out 
                # each line adds each sequence from the line 
                for line in infile:
                    threeVals=line.split(",")
                    threeVals[2]=threeVals[2].replace("\n","")
                    seqs[index].append(threeVals)
    
    # not going to worry about repeats for now
    cddict={}
    for cd in seqs:
        for seq in cd:
            #import pdb; pdb.set_trace()
            cddict[seq[2][:-3]]=seq[:2]
    
                    
    
    return cddict


def addCDextra(seqs, vj, cddict):
    """
    this takes the CDR3 and adds the other CDRs based on the dictionary generated
    by extra cds
    """
    newSeq=[]
    newV=[]
    fail=0
    for idx, seq in enumerate(seqs):
        try:
            #import pdb; pdb.set_trace()
            cds=cddict[seq]
            # cds should be [cdr1, cdr2]
            # limit it to size [5,6] as there are a few CDR2s that are funky sizes
            if len(cds[0])==5 and len(cds[1])==6:
                newSeq.append(seq+cds[0]+cds[1])
                newV.append(vj[idx][0])
        except:
            fail+=1
    #print("Failures: {}".format(fail))
    return newSeq, newV

def subNames(file, c):
    """
    Subroutine for getVJnames() that actually reads the file and makes the dict
    c - the chain 
    """
    dic={}
    with open(file, 'r') as jf:
        for idx, line in enumerate(jf):
            line=line.split("|")
            # we only care about the second item
            line=line[1] # TRBJ1-1
            line=line.split("-") #[TRBJ1,1]
            first=line[0]
            # from here the the code can have two values e.g. TRBJ11
            if "/" in first: # sorts out examples like this: TRAV29/DV5
                first = first.split('/')
                first = first[0]
            # sort out double digits if they exist
            first=first[-2:] # gets last two digits
            try:
                first=int(first) # covers double values
            except:
                try:
                    first=int(first[-1]) # last of two dig is gonna be an int apart
                except:
                    first=69 # lol
            try:#make sure it exists
                second=line[1]
                if len(second)>1: # can look like 2P
                    try:
                        second=int(second)
                    except:
                        second=int(second[0])
                else:
                    second=int(second)
            except:
                second=0
            line=[first, second] #[1,1]
            # this leaves [family, subfamily]
            dic[idx]=line
    return dic
          
def getVJnames(chain='beta', path='data/tags/'):
    """
    This generates a dictionary of the family and subfamily of v and j genes
    """
    init=path+'human_extended_TR'
    if chain=='beta':
        init=init+'B'
    else:
        init=init+'A'
    jFile=init+'J.tags'
    vFile=init+'V.tags'
    """
    format of tag files is e.g.          
    GGACAAGGCACCAGACTCAC 20 K02545|TRBJ1-1|Homo
    """
    jd=subNames(jFile, 'j')
    vd=subNames(vFile, 'v')
    
    return jd, vd
    
    
    
    
    
    

    
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 11:34:07 2017

@author: lewismoffat

This script takes the dcr decombinated files and the retrieved names of V and J
Genes and then performs analysis. The first thing it does is look at the histograms
of V and J use

From there we choose one specific V region (starting with a more common one)
and plot the relative use of different tuples 


"""

########################################################
# Module Imports 
########################################################
import numpy as np
import dataProcessing as dp
import pdb
import sklearn as sk
import matplotlib.pyplot as plt
import pandas
from collections import Counter
from operator import itemgetter
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier as KNN
import glob
from sklearn.manifold import TSNE

from matplotlib import pylab

########################################################
# Parameters
########################################################


# File names for different data; A - alpha chain, B - beta chain            
cd4A_file = 'patient1/vDCRe_alpha_EG10_CD4_naive_alpha.dcrcdr3'
cd8A_file = 'patient1/vDCRe_alpha_EG10_CD8_naive_alpha.dcrcdr3'
cd4B_file = 'patient1/vDCRe_beta_EG10_CD4_naive_beta.dcrcdr3'
cd8B_file = 'patient1/vDCRe_beta_EG10_CD8_naive_beta.dcrcdr3'
data = 'data/'
extra = 'VandJ/'

graphs=False
files = [cd4B_file, cd8B_file]

########################################################
# Data Retrieval 
########################################################
"""
The data for the first patient is stored in the 'data/patient1' file
where each sequence is encoded as a string and comma separated with its count
at current extraction is unique and count is ignored
"""

#files=glob.glob("F:/seqs/*.txt")
#cd4, cd8 = dp.dataReader(files, ["naive","beta"])


# sequence list to be filled. Make sure file order is the same as a below
cd4=[]
cd8=[]
seqs=[cd4,cd8]   # this contains the sequences 

cd4vj=[]
cd8vj=[]
vj=[cd4vj,cd8vj] # this contains the v and j index genes




for index, file in enumerate(files):
    file=data+extra+file
    with open(file,'r') as infile:
        # goes through each of the files specified in read mode and pulls out 
        # each line and formats it so a list gets X copies of the sequence 
        for line in infile:
            twoVals=line.split(":")
            twoVals[1]=twoVals[1].replace("\n","")
            twoVals[1]=twoVals[1].split(",")
            twoVals[0]=twoVals[0].split(",")
            seqs[index].append(twoVals[1][0])
            vj[index].append(twoVals[0][:2])
            
# data is in the format [v_index, j_index, deletionsV, deletionsJ, extra:CDR3, count]


#### Run the CD4 analysis first 
v=[]
j=[]
CD4=0
CD8=1

# SPECIFY HERE
CD=CD8


for row in vj[CD]:
    v.append(int(row[0]))
    j.append(int(row[1]))
    

if graphs:
    # plot the v gene usage 
    # convert list to numpy array
    lenArr=np.asarray(v) 
    
    # bins
    binNum=max(lenArr)-min(lenArr)
    bins=np.arange(min(lenArr),max(lenArr))
    
    # setup graph
    plt.figure(figsize=(15,15))
    plt.hist(lenArr, bins=binNum)
    plt.xticks(bins)
    plt.title("V Gene Usage in CD8")
    plt.xlabel("V Gene Index")
    plt.ylabel("Frequency")
    plt.show()
    
    
    # plot the v gene usage 
    # convert list to numpy array
    lenArr=np.asarray(j) 
    
    # bins
    binNum=max(lenArr)-min(lenArr)
    bins=np.arange(min(lenArr),max(lenArr))
    
    # setup graph
    plt.figure(figsize=(15,15))
    plt.hist(lenArr, bins=binNum)
    plt.xticks(bins)
    plt.title("J Gene Usage in CD8")
    plt.xlabel("J Gene Index")
    plt.ylabel("Frequency")
    plt.show()
    

#df = pandas.DataFrame.from_dict(letter_counts, orient='index')
#df.plot(kind='bar')
#ax1 = plt.axes()
#x_axis = ax1.axes.get_xaxis()
#x_axis.set_visible(False)
#plt.show()
#plt.close()



# Chose 1 specific V region (start with a more common one) and then plot the 
# use of each triplet in the CDR3 in a sample of CD4 and CD8 cells (ranked 
# according to differential usage ?).
#==============================================================================
 
#==============================================================================
Vcounter=Counter(v)
print(Vcounter.most_common(2)[1])
# for CD4 this is 15 : TRBV20-1|Homo
# for CD8 this is 6  : TRBV12-4|Homo  (this is second most common for CD4)
filtered=[]
for idx, row in enumerate(vj[CD]):
    if int(row[0])==int(Vcounter.most_common(2)[1][0]):
        filtered.append(idx)
        
# filtered now contains a list of index locations to retrieve the CDR3 sequences
# for the most common v region
filtSeq=[]
for idx in filtered:
    filtSeq.append(seqs[CD][idx])

tuples=[]
for seq in filtSeq:
    ptup=dp.pTuple(seq,n=3)
    for tup in ptup:
        tuples.append(tup)
        

letter_counts_4 = Counter(tuples)
#df = pandas.DataFrame.from_dict(letter_counts, orient='index')
#df.plot(kind='bar')

### Run the CD8 analysis
CD=CD8
v=[]
j=[]
for row in vj[CD]:
    v.append(int(row[0]))
    j.append(int(row[1]))

Vcounter=Counter(v)
print(Vcounter.most_common(1))
# for CD4 this is 15 : TRBV20-1|Homo
# for CD8 this is 6  : TRBV12-4|Homo
filtered=[]
for idx, row in enumerate(vj[CD8]):
    if int(row[0])==int(Vcounter.most_common(1)[0][0]):
        filtered.append(idx)
        
# filtered now contains a list of index locations to retrieve the CDR3 sequences
# for the most common v region
filtSeq=[]
for idx in filtered:
    filtSeq.append(seqs[CD8][idx])

tuples=[]
for seq in filtSeq:
    ptup=dp.pTuple(seq,n=3)
    for tup in ptup:
        tuples.append(tup)
        
letter_counts_8 = Counter(tuples)

difs=[]

# this gets the difference between counts of the two things
#allKeys=letter_counts_8.keys()+letter_counts_4.keys()


for key in letter_counts_8.keys():
    eight_count=letter_counts_8[str(key)]
    four_count=letter_counts_4[str(key)]
    difference=eight_count-four_count
    if difference<0:
        difference=difference*-1
    difs.append([key,difference])

#stand=Counter()
#for val in difs:
difs=sorted(difs, key=itemgetter(1), reverse=True)




#==============================================================================
# I suggest in the first place we do some semi-qualitative “inspection”. Take 
# an individual V and J region combination, and focus on those. I would plot 
# the proportion of amino acids at each position for these, and see if we start
# to see any differences between the classes. To deal with different lengths, 
# you could either take one class length at a time; or bin the positions into a
# 1:15 vector (i.e. calculate a position as a proportion of length, and then 
# assign to one of the 12:15 bins). The idea is to look carefully at a subset 
# of data and see if anything emerges to the eye. We can then use this to build
# a more systematic classifier.
#==============================================================================


# Filter out the most common VJ combo and then get the corresponding CDR3 for 
# CD4
CD=CD4
vandj=[]
for row in vj[CD]:
    vandj.append(str(row))
letter_counts_4 = Counter(vandj)
filtvandj=[]
print(letter_counts_4.most_common(1)[0][0])
for idx, row in enumerate(vandj):
    if str(row)==letter_counts_4.most_common(1)[0][0]:
        filtvandj.append(idx)
seqs4=[]
for idx in filtvandj:
    seqs4.append(seqs[CD][idx])


# Filter out the most common VJ combo and then get the corresponding CDR3 for 
# CD8
        
CD=CD8
vandj=[]
for row in vj[CD]:
    vandj.append(str(row))
letter_counts_8 = Counter(vandj)
filtvandj=[]
print(letter_counts_8.most_common(1)[0][0])
for idx, row in enumerate(vandj):
    if str(row)==letter_counts_4.most_common(1)[0][0]:
        filtvandj.append(idx)

seqs8=[]
for idx in filtvandj:
    seqs8.append(seqs[CD][idx])

    
# Filter the two seqs to only get the 14 long seqs
seqs4_14=[]
seqs8_14=[]
for seq in seqs4:
    #if len(seq)<17 and len(seq)>11:
    if len(seq)==14: 
        seqs4_14.append(seq)
for seq in seqs8:
    if len(seq)==14:
    #if len(seq)<17 and len(seq)>11:
        seqs8_14.append(seq)
seqs4_14_=seqs4_14.copy()
seqs8_14_=seqs8_14.copy()







#==============================================================================
# tSNE within a VJ combo region
#==============================================================================



seqs4_14_=dp.seq2fatch(seqs4_14_) 
seqs8_14_=dp.seq2fatch(seqs8_14_) 


tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)

# use function to create data
X, y = dp.dataCreator(seqs4_14_,seqs8_14_)

# shuffle data
X, y = sk.utils.shuffle(X,y)

two_d_embeddings_1 = tsne.fit_transform(X)

def plot(embeddings, labels):
    assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
    pylab.figure(figsize=(15,15))  # in inches
    for i, label in enumerate(labels):
        x, y = embeddings[i,:]

        if label>0.5:
            pylab.scatter(x, y, c='b')
        else:
            pylab.scatter(x, y, c='r')
        #pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',  ha='right', va='bottom')
    
    pylab.show()
    return
    
plot(two_d_embeddings_1, y)

pdb.set_trace()




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
   
              
# Function to replace characters with numbers
def seq2num(seqs, diction):
    seqsNew=[]
    for seq in seqs:
        short=[]
        [short.append(diction[char]) for char in seq]
        seqsNew.append(short)
    
    return np.array(seqsNew)
    
seqs8_14=seq2num(seqs8_14,intDictIndex)
seqs4_14=seq2num(seqs4_14,intDictIndex)

# now need to get a list of [AA, postion]


def coordinator(seqs):
    coord=[]
    for seq in seqs:
        for idx, aa in enumerate(seq):
            coord.append([aa, idx])
    return np.array(coord)
            

seqs8_14=coordinator(seqs8_14)
seqs4_14=coordinator(seqs4_14)

### This is for plotting the 3D bar graph of the CD8 AA usage
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x=seqs8_14[:,0]
y=seqs8_14[:,1]

hist, xedges, yedges = np.histogram2d(x, y, bins=20, range=[[0, 20], [0, 14]])

    
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
xpos = xpos.flatten('F')
ypos = ypos.flatten('F')
zpos = np.zeros_like(xpos)

# Construct arrays with the dimensions for the 16 bars.
dx = 0.5 * np.ones_like(zpos)
dy = dx.copy()
dz = hist.flatten()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
ax.set_title('')

ax.set_title('CD8 AA Usage for 14 Long')
plt.show()




### This is for plotting the 3D bar graph of the CD4 AA usage
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x=seqs4_14[:,0]
y=seqs4_14[:,1]

hist1, xedges, yedges = np.histogram2d(x, y, bins=20, range=[[0, 20], [0, 14]])

    
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
xpos = xpos.flatten('F')
ypos = ypos.flatten('F')
zpos = np.zeros_like(xpos)

# Construct arrays with the dimensions for the 16 bars.
dx = 0.5 * np.ones_like(zpos)
dy = dx.copy()
dz = hist1.flatten()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
ax.set_title('CD4 AA Usage for 14 Long')
plt.show()


### This is for plotting the 3D bar graph of the Difference
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x=seqs4_14[:,0]
y=seqs4_14[:,1]

#hist, xedges, yedges = np.histogram2d(x, y, bins=20, range=[[0, 20], [0, 14]])
hist=np.abs(hist-hist1)
    
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
xpos = xpos.flatten('F')
ypos = ypos.flatten('F')
zpos = np.zeros_like(xpos)

# Construct arrays with the dimensions for the 16 bars.
dx = 0.5 * np.ones_like(zpos)
dy = dx.copy()
dz = hist.flatten()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
ax.set_title('Difference')
plt.show()


#==============================================================================
# Classification using KNN 
#==============================================================================

seqs4_14_2=seqs4_14_.copy()
seqs8_14_2=seqs8_14_.copy()

cd4=dp.seq2fatch(seqs4_14_)
cd8=dp.seq2fatch(seqs8_14_)


# from this point it is assumed cd4/8 are NUMPY vectors
# labels are created and then the X and Y vectors are shuffled and combo'd
y4=np.zeros((len(cd4)))
y8=np.zeros((len(cd8)))
y4[:]=0
y8[:]=1

# combine classes
Y = np.concatenate((y4,y8),0)
X = np.concatenate((cd4,cd8),0)

print("CD4 to CD8 Ratio {}:{}".format(len(cd4)/(len(cd4)+len(cd8)),len(cd8)/(len(cd4)+len(cd8))))
print("Total Sequences: {}".format(len(cd4)+len(cd8)))
# memory clean up
cd4=None
cd8=None
y4=None
y8=None


X, Y = sk.utils.shuffle(X,Y)
xTrain, xHalf, yTrain, yHalf = train_test_split(X, Y, test_size=0.20) 
# memory clean up
X=None
Y=None 

xVal, xTest, yVal, yTest= train_test_split(xHalf, yHalf, test_size=0.50) 

# Some memory clean up
xHalf=None
yHalf=None
sqHalf=None


print("Data Loaded and Ready...")
print("Training K-NN")
neigh=KNN(n_neighbors=1)
neigh.fit(xTrain, yTrain)
y_true, y_pred = yVal, neigh.predict(xVal)
print("{} Validaton Accuracy".format(accuracy_score(y_true, y_pred)))
print(classification_report(y_true, y_pred))

#==============================================================================
# P-Tuple Analysis
#==============================================================================

both=[seqs4_14_2,seqs8_14_2]

for seqs in both:
    for idx, seq in enumerate(seqs):
        ptup=dp.pTuple(seq,n=1)
        seqs[idx]=ptup
# flattens list
shrtCd4s = [item for sublist in seqs4_14_2 for item in sublist]    
# flattens list
shrtCd8s = [item for sublist in seqs8_14_2 for item in sublist]        

#print("Number of Tuples: {}".format(len(shrtCd4)+len(shrtCd8)))
cd4Count=Counter(shrtCd4s)
cd8Count=Counter(shrtCd8s)

fiveMostCommon1=np.array(cd4Count.most_common(n=20))
print(fiveMostCommon1)  

fiveMostCommon2=np.array(cd8Count.most_common(n=20))
print(fiveMostCommon2)



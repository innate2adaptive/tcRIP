# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 15:07:43 2017

@author: lewismoffat
"""
#==============================================================================
# Imports
#==============================================================================

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC

import dataProcessing as dp

#==============================================================================
# Parameters
#==============================================================================

# File names for different data; A - alpha chain, B - beta chain            
cd4A_file = 'patient1/vDCRe_alpha_EG10_CD4_naive_alpha.dcrcdr3'
cd8A_file = 'patient1/vDCRe_alpha_EG10_CD8_naive_alpha.dcrcdr3'
cd4B_file = 'patient1/vDCRe_beta_EG10_CD4_naive_beta.dcrcdr3'
cd8B_file = 'patient1/vDCRe_beta_EG10_CD8_naive_beta.dcrcdr3'
data = 'data/'
extra = 'VandJ/'



#==============================================================================
# Read Data Files
#==============================================================================

# Files to be read
files = [cd4B_file, cd8B_file]

# sequence list to be filled. Make sure file order is the same as a below
cd4=[]
cd8=[]
seqs=[cd4,cd8]   # this contains the sequences 
cd4vj=[]
cd8vj=[]
vj=[cd4vj,cd8vj] # this contains the v and j index genes
ex4=[]
ex8=[]
extraVals=[ex4, ex8]


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
            vj[index].append(twoVals[0][:1])
            extraVals[index].append(twoVals[0][:4])

            
# filter out joint sequences
seqs[0], seqs[1], vj[0], vj[1], joint = dp.removeDup(seqs[0], seqs[1], vj[0], vj[1])

# add extrac cds
cddict=dp.extraCDs()
seqs[0], vj[0]=dp.addCDextra(seqs[0],vj[0],cddict)
seqs[1], vj[1]=dp.addCDextra(seqs[1],vj[1],cddict)

for idx, seqz in enumerate(seqs):
    for idx2, seq in enumerate(seqz):
        if idx==1:
            seqs[idx][idx2]=['CD8',vj[idx][idx2],seq[-5:],seq[-11:-5],seq[:-11]]           
        if idx==0:
            seqs[idx][idx2]=['CD4',vj[idx][idx2],seq[-5:],seq[-11:-5],seq[:-11]]
            
            
#==============================================================================
# Create Seq Records          
#==============================================================================

r1=[]
r2=[]
records=[r1,r2]            

for idx, seqz in enumerate(seqs):
    for idx2, seq in enumerate(seqz):
        newID=seqs[idx][idx2][1]+"_" + seqs[idx][idx2][2]+"_"+seqs[idx][idx2][3]
        record = SeqRecord(Seq(seqs[idx][idx2][-1],
                               IUPAC.protein),
                           id=newID, name=newID,
                           description=newID)
        records[idx].append(record)
        


##==============================================================================
## Writing the fasta file
##==============================================================================


with open("cd4.fasta", "w") as output_handle:
    SeqIO.write(records[0], output_handle, "fasta")
with open("cd8.fasta", "w") as output_handle:
    SeqIO.write(records[1], output_handle, "fasta")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 11:44:33 2017

@author: lewismoffat


This takes the freq files and cdr3 files for the individual patients and
combines them into one text file that contains the extra v and j gene info
specifically v region, j region, and the respective deletions, and the added sequence
"""
import glob

filesFrq=glob.glob("F:/seqs/*.freq")
filesCD=glob.glob("F:/seqs/*.cdr3")

# use the freq files a cycle 
for fil in filesFrq:
    for filCD in filesCD:
        if fil.replace(".freq","")==filCD.replace(".cdr3",""): # if the files match
            # create new file name
            filnew=fil.replace(".freq","")+".txt"
            file = open(filnew,"w")
            # open both cdr3 and freq
            with open(filCD, "r") as cd:
                with open(fil, "r") as fr:
                    for line1, line in zip(fr,cd):
                        file.write(line1.replace("\n","")+", "+line)
            file.close()
            

            
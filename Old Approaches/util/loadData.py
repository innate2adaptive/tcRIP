# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 12:51:25 2017

@author: lewismoffat
"""

import numpy as np

# read the file for alpha chain CD4
cd4A_file = open('data/patient1/vDCRe_alpha_EG10_CD4_naive_alpha.txt', 'r')
cd8A_file = open('data/patient1/vDCRe_alpha_EG10_CD8_naive_alpha.txt', 'r')
cd4B_file = open('data/patient1/vDCRe_beta_EG10_CD4_naive_beta.txt', 'r')
cd8B_file = open('data/patient1/vDCRe_beta_EG10_CD8_naive_beta.txt', 'r')

print('Read Files...')

# lists to contain [sequence, number]
cd4A=list()
cd8A=list()
cd4B=list()
cd8B=list()

# fill and format arrays
for line in cd4A_file:
    twoVals=line.split(", ")
    twoVals[1]=twoVals[1].replace("\n","")
    cd4A.append(twoVals)
for line in cd8A_file:
    twoVals=line.split(", ")
    twoVals[1]=twoVals[1].replace("\n","")
    cd8A.append(twoVals)
for line in cd4B_file:
    twoVals=line.split(", ")
    twoVals[1]=twoVals[1].replace("\n","")
    cd4B.append(twoVals)
for line in cd8B_file:
    twoVals=line.split(", ")
    twoVals[1]=twoVals[1].replace("\n","")
    cd8B.append(twoVals)

print('Arrays Ready...')
    
# convert to numpy arrays
cd4A=np.asarray(cd4A)
cd8A=np.asarray(cd8A)
cd4B=np.asarray(cd4B)
cd8B=np.asarray(cd8B)

np.save('cd4A',cd4A)
np.save('cd8A',cd8A)
np.save('cd4B',cd4B)
np.save('cd8B',cd8B)

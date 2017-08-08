# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 14:25:08 2017

@author: lewismoffat


This script is statistics focused. It generates the general plots and values
for the overall dataset.

It is expected that the data is located in the directory: F:/seqs/*.cdr

"""

#==============================================================================
# Module Imports 
#==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import dataProcessing as dp
import pdb

from collections import defaultdict
from collections import Counter


#==============================================================================
# Get the data
#==============================================================================

# are we doing the full set
singlePatient=False

# which patient to get data from 
patient=['Complete']
chain = "beta"


if singlePatient:
    print('Patient: '+patient[0])
    delim = ["naive",chain]+patient #other delimiters
else:
    print('Patient: All')
    delim = ["naive",chain] #other delimiters


seqs, vj = dp.loadAll(delim) # these gets all the sequences and vj values

#==============================================================================
# Basic Analytics
#==============================================================================

# filter out joint sequences
seqs[0], seqs[1], vj[0], vj[1], joint = dp.removeDup(seqs[0], seqs[1], vj[0], vj[1])

# use function to create data
X, y = dp.dataCreator(seqs[0],seqs[1])

# print class balances
dp.printClassBalanceV2(seqs[0],seqs[1])


#==============================================================================
# Sequence Venn Diagram
#==============================================================================
dp.venn(len(seqs[0]), len(seqs[1]), len(joint))


#==============================================================================
# Sequence length histograms
#==============================================================================
dp.lenHisto(seqs[0], seqs[1], Title=("for"+" "+patient+" "+"Dataset"))


























# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 23:59:01 2017

@author: lewismoffat
"""

with open('F:/beta07.fastq') as f:
    counter=0
    for line in f:
        counter+=1
        print(line)
        if counter>3:
            import pdb; pdb.set_trace()
    
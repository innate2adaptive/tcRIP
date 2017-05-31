# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 11:46:56 2017

@author: lewismoffat

Contains the Atchely factors and the corresponing function to get the five 
factors for  a given AA through use of a default dict

"""
import numpy

#                               F1      F2      F3      F4      F5
atchFactDict={'A':numpy.array([-0.591, -1.302, -0.733,  1.57,  -0.146]),
              'C':numpy.array([-1.343,  0.465, -0.862, -1.02,  -0.255]),
              'D':numpy.array([ 1.05,   0.302, -3.656, -0.259, -3.242]),
              'E':numpy.array([ 1.357, -1.453,  1.477,  0.113, -0.837]),
              'F':numpy.array([-1.006, -0.59,  1.891, -0.397,  0.412]),
              'G':numpy.array([-0.384,  1.652,  1.33,   1.045,  2.064]),
              'H':numpy.array([ 0.336, -0.417, -1.673, -1.474, -0.078]),
              'I':numpy.array([-1.239, -0.547,  2.131,  0.393,  0.816]),
              'K':numpy.array([ 1.831, -0.561,  0.533, -0.277,  1.648]),
              'L':numpy.array([-1.019, -0.987, -1.505,  1.266, -0.912]),
              'M':numpy.array([-0.663, -1.524,  2.219, -1.005,  1.212]),
              'N':numpy.array([ 0.945,  0.828,  1.299, -0.169,  0.933]),
              'P':numpy.array([ 0.189,  2.081, -1.628,  0.421, -1.392]),
              'Q':numpy.array([ 0.931, -0.179, -3.005, -0.503, -1.853]),
              'R':numpy.array([ 1.538, -0.055,  1.502,  0.44,   2.897]),
              'S':numpy.array([-0.228,  1.399, -4.76,   0.67,  -2.647]),
              'T':numpy.array([-0.032,  0.326,  2.213,  0.908,  1.313]),
              'V':numpy.array([-1.337, -0.279, -0.544,  1.242, -1.262]),
              'W':numpy.array([-0.595,  0.009,  0.672, -2.128, -0.184]),
              'Y':numpy.array([ 0.26,   0.83,   3.097, -0.838,  1.512])}





def atchleyFactor(AA):
    """
    In:  string, amino acid single letter code
    Out: (1,5) numpy array, gives the five atchely Factors
    """
    return atchFactDict[AA] 
    
"""
As per Atchley et. al. (2005):

Factor I is bipolar (large positive and negative factor coefficients) and reflects simultaneous covariation in portion of exposed residues versus buried residues, nonbonded energy versus free energy, number of hydrogen bond donors, polarity versus nonpolarity, and hydrophobicity versus hydrophilicity (Table 1). The variable with the largest positive value was average nonbonded energy per atom computed as an average of pairwise interactions between constituent atoms (20). Another energy variable with a large negative coefficient is transfer free energy, which represents the partition coefficient of a particular amino acid between buried and accessible molar fractions (21). For simplicity, we designate Factor I as a polarity index. Table 5 shows those attributes from the original database with high pairwise product moment correlation coefficients (>0.85) with Factor I.

Factor II is a secondary structure factor. There is an inverse relationship of relative propensity for various amino acids in various secondary structural configurations, such as a coil, a turn, or a bend versus the frequency in an α-helix.

Factor III relates to molecular size or volume with high factor coefficients for bulkiness, residue volume, average volume of a buried residue, side chain volume, and molecular weight. A large negative factor coefficient occurs for normalized frequency of a left-handed α-helix.

Factor IV reflects relative amino acid composition in various proteins, number of codons coding for an amino acid, and amino acid composition. These attributes vary inversely with refractivity (22) and heat capacity (23).

Factor V refers to electrostatic charge with high coefficients on isoelectric point and net charge. As expected, there is an inverse relationship between positive and negative charge.

"""
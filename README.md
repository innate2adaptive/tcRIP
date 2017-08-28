# TCRIP: T-Cell Receptor Information Prediction

This is the code for the MSc Thesis: "Classification of TCR&beta; Repertoire between CD4<sup>+</sup> and CD8<sup>+</sup> T-Cell Populations"

This code has been written by Lewis Iain Moffat in partial completion of the MSc Machine Learning awarded by University College London, Department of Computer Science. This project used a Machine Learning approach to classifying T-Cell Receptor data as either having come from CD4 or CD8 T-Cell lineages. This is done through a combination of feature engineering methods classification algorithms.

## Getting Started
This project does not require installation and only needs to be cloned from the repository. Run the following command to get a copy of the project up and running on your local machine. Make sure you have git installed.

```git
git clone https://github.com/Groovy-Dragon/tcRIP
```
 

### Prerequisites

These scripts use python 3.5 and a variety of libraries. The following are for general usage of the scripts:

<ul>
    <li> Biopython 1.68 </li>
    <li> nltk 3.2.1 </li>
    <li> numpy 1.12.1 </li>
    <li> pandas 0.19.2 </li>
    <li> scipy 0.18.1 </li>
    <li> tqdm 4.11.2 - this is not entirely neccessary and can run without in most scripts</li> 
</ul>

These libraries are for the various plot generation functions and scripts:

<ul>
    <li> Matplotlib 2.0.0 </li>
    <li> Matplotlib-Venn 0.11.5 </li>
    <li> MPLD3 0.2 </li>
    <li> Seaborn 3.4.0.3 </li> 
</ul>

These libraries are specifically for the machine learning operations like feature engineering and classification:

<ul>
    <li> Edward 1.3.2 </li>
    <li> Imbalanced-Learn 0.2.1 </li>
    <li> Keras 2.0.4 </li>
    <li> Scikit-Learn 0.18.1 </li>
    <li> Tensorflow 1.1.0 </li>
    <li> XGBoost 0.6 </li>
</ul>

### File Names
Due to the modular nature of this work the majority of the scripts are standalone. 
NEED TO COMPLETE THIS


## Running the Code

The following section describes the basics of using some of the underlying functions.

### Load the Data
First import the [data processing](../blob/master/dataProcessing.py) script which contains a large number of data loaders, feature engineering functions, and conveniance functions. 

```python
import dataProcessing as dp
```

Now use the following code block to load the sequences and filter out the CDR3s that are in both classes. Note that this will load the data from the data records file (../blob/master/data_records). Note that although the V and J genes are provided in separate lists they match to the sequences by index. 
```python


# load in sequences and vj info for all patients with naive and beta chains
# if you want to get specific patient data add the patient name as a string to the list argument e.g. 'KS07', 'SK11', 'EG10'
seqs, vj = dp.loadAllPatients(['naive','beta']) 

# filter out joint sequences
seqs[0], seqs[1], vj[0], vj[1], joint = dp.removeDup(seqs[0], seqs[1], vj[0], vj[1])
# seqs[0] contains the CD4 sequences, and seqs[1] contains the CD8; vice versa with the v and j gene indexes in vj
# joint contains the sequences that are shared

# print the number of shared sequences as a number and percent
print("Number of Shared Seqs: {}".format(len(joint)))
print("Shared Percent: %.2f%%" % (len(joint)/(len(seqs[0])+len(seqs[1])) * 100.0))
```

If you want to load in the CDR1 and CDR2 sequences it is done by loading a dictionary which when called extracts the CDR1 and CDR2 from data files. Note that some of the CDR1s are 5 amino acids long, compared to the vast majority which are 6 amino acids long. Due to length invariance this very small number is filtered out.

```python
# add extrac cds by getting the dicitionary
cddict=dp.extraCDs()
# use the dictionary to add the sequences together. This concatenates the CDR1 and CDR2 to the CDR3 in one big string. This string is then the CDR3+CDR2+CDR1. 
seqs[0], vj[0]=dp.addCDextra(seqs[0],vj[0],cddict)
seqs[1], vj[1]=dp.addCDextra(seqs[1],vj[1],cddict)
```
From this point you are free to use your own feature engineering or the provided functions. 



### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Contributing

Please feel free to submit a pull requests.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Lewis Moffat** - *Initial work* - [Groovy-Dragon](https://github.com/Groovy-Dragon)

See also the list of [contributors](https://github.com/Groovy-Dragon/tcRIP/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details

## Acknowledgments

* My Primary Supervisor: Prof. Benny Chain, UCL Division of Immunology, [Innate2Adaptive](http://www.innate2adaptive.com/)
* My Secondary Supervisro: Prof. John Shawe-Talyor, [Head of UCL Computer Science Department](http://www0.cs.ucl.ac.uk/staff/j.shawe-taylor/)

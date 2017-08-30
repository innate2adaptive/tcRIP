# TCRIP: T-Cell Receptor Information Prediction

This is the code for the MSc Thesis: "Classification of TCR&beta; Repertoire between CD4<sup>+</sup> and CD8<sup>+</sup> T-Cell Populations"

This code has been written by Lewis Iain Moffat in partial completion of the MSc Machine Learning awarded by University College London, Department of Computer Science. This project used a Machine Learning approach to classifying T-Cell Receptor data as either having come from CD4 or CD8 T-Cell lineages. This is done through a combination of feature engineering methods classification algorithms.

## Getting Started
This project does not require installation and only needs to be cloned from the repository. Run the following command to get a copy of the project up and running on your local machine. Make sure you have git installed.

```git
git clone https://github.com/Groovy-Dragon/tcRIP
```
 

### Prerequisites

These scripts use python 3.6 and a variety of libraries. The following are for general usage of the scripts:

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
Due to the modular nature of this work the majority of the scripts are standalone. The key file that contains utility methods is:
```
dataProcessing.py
```
This mostly contains feature engineering utility functions, dataset creation functions, and others. The rest of the functions are generally modular in nature and perform a different classification or data exploration experiment. <b>See the wiki for a short description of each file and what it does</b>. 


## Running the Code

The following section describes the basics of using some of the underlying functions.

### Load the Data
First import the [data processing](../blob/master/dataProcessing.py) script which contains a large number of data loaders, feature engineering functions, and conveniance functions. Also load Sci-Kit learn.

```python
import dataProcessing as dp
import sklearn as sk
```

Now use the following code block to load the sequences and filter out the CDR3s that are in both classes. Note that this will load the data from the [data records file](../blob/master/data_records). Note that although the V and J genes are provided in separate lists they match to the sequences by index. 
```python


# load in sequences and vj info for all patients with naive and beta chains
# if you want to get specific patient data add the patient name as a string to the list argument e.g. 'HV1', 'HV2', or 'HV3'
seqs, vj = dp.loadAllPatients(['naive','beta']) 

# filter out joint sequences
seqs[0], seqs[1], vj[0], vj[1], joint = dp.removeDup(seqs[0], seqs[1], vj[0], vj[1])
# seqs[0] contains the CD4 sequences, and seqs[1] contains the CD8; vice versa with the v and j gene indexes in vj
# joint contains the sequences that are shared

# print the number of shared sequences as a number and percent
print("Number of Shared Seqs: {}".format(len(joint)))
print("Shared Percent: %.2f%%" % (len(joint)/(len(seqs[0])+len(seqs[1])) * 100.0))
```

If you want to load in the CDR1 and CDR2 sequences it is done by loading a dictionary which when called extracts the CDR1 and CDR2 from data files. Note that some of the CDR1s are 5 amino acids long, compared to the vast majority which are 6 amino acids long. Due to length invariance this very small number is filtered out. Also the classes have been kept separate up until shuffling and performing the train test split. 

```python
# add extrac cds by getting the dicitionary
cddict=dp.extraCDs()
# use the dictionary to add the sequences together. This concatenates the CDR1 and CDR2 to the CDR3 in one big string. This string is then the CDR3+CDR2+CDR1. 
seqs[0], vj[0]=dp.addCDextra(seqs[0],vj[0],cddict)
seqs[1], vj[1]=dp.addCDextra(seqs[1],vj[1],cddict)
```
From this point can use your own feature engineering or the provided functions. 

### Feature Engineering

This work contains several quick functions for converting CDR strings in a list to a numerical format i.e. feature engineering. 
One of these methods is converting sequences to Atchley Vectors. 

```python
# filter the sequences to only get CDRs of a set length e.g. 14
seqs[0]=dp.filtr(seqs[0], 14)
seqs[1]=dp.filtr(seqs[1], 14)

# This takes the lists of CDRs (e.g. ['CASSADL..','CSARDFS...',...,'CASRDFSG...']) and converts them to flat atchley vectorized 
# sequences. 

seqs[0]=dp.seq2fatch(seqs[0])
seqs[1]=dp.seq2fatch(seqs[1])
    
```
An example of full classification using this approach is in the [Li et al Script](../blob/master/ML_Li.py)

Another method is generating p-Tuple vectors, or frequency counts of p-long subsequences of amino acids from each CDR3.

```python
# set the tuple lenth
tuplen=3

# Run the tuple calculation. 
seqs[0]=dp.char2ptuple(seqs[0], n=tuplen)
seqs[1]=dp.char2ptuple(seqs[1], n=tuplen)
```
You'll notice that the output of these vectors is a scipy conpressed matrix. All of the Sci-Kit learn classifiers and XGBoost can handle it but for ease it is better to convert it from COO to CDR format. This allows slicing. This is done like this:

```python
# Convert to CSR 
seqs[0]=seqs[0].tocsr()
seqs[1]=seqs[1].tocsr()
```
For an example of p-Tuple classification see the [p-Tuple script](../blob/master/ML_Ptuple.py). Another method for feature engineering is using protein embeddings vectors. These embeddings are either the SwissProt trained methods provided by [Asgari and Mofrad](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0141287) (2015) or the custom embeddings trained on this dataset. New embeddings can also be trained using the [ProtVec script](../blob/master/UL_ProtVec.py). 
```python
# Are we using the SwissProt embeddings? If not, it defaults to custom embeddings
swiss=True

# convert the strings to X dimensional vectors from the embeddings
seqs[0]=dp.GloVe(seqs[0], True)
seqs[1]=dp.GloVe(seqs[1], True)
```
The data processing script contains several more of these methods. These will be written about further in the future in the wiki.

### Dataset Pre-Processing

Once the two classes have been converted to consistent length numeric values by feature engineering, we can use some utility functions to build our 'X' and 'y' pairs. Where 'X' is the data and 'y' are the labels. 

```python
# get an sklearn utility function 
from sklearn.model_selection import train_test_split 

# use function to create data
# this expects arguments of CD4, CD8 and labels them 0, 1 respectively. 
X, y = dp.dataCreator(seqs[0],seqs[1])

# shuffle data using Sci-Kit Learn function
X, y = sk.utils.shuffle(X,y)

# print class balances
dp.printClassBalance(y)

# 20% Validation set
xTrain, xVal, yTrain, yVal= train_test_split(X, y, test_size=0.20) 
```
You'll notice that the test set is not included. While writing the paper the test set was set aside in separate files however here this has been left to the user and all data is provided when loaded. 

### Classification
Once the data has been processed it can now be used for training and testing a classification algorithm. A single example is as follows using an Adaboost classifier.

```python
from sklearn.ensemble import AdaBoostClassifier

# Set up an adaboost Classifer
clf = AdaBoostClassifier(n_estimators=100)

# Fit the booster
clf.fit(xTrain, yTrain)
```
An alternative to fitting the model is to fit the model by grid search around hyper-parameters. An example is given below using an SVM classifier, where the C value, kernal, and Gamma (depending on kernel) value are optimized. Note that this will take a long time as it also performs cross-validation. So for each of k folds, there will be a however many grid search combinations to train.

```python
from sklearn.svm import SVC

# set grid search across best parameters for both Linear and RBF kernels
tuned_parameters = [{'kernel': ['rbf','linear'], 'C': [0.01,0.1,1,10,100], 'gamma':[1e-5,1e-4,1e-3,1e-2,1e-1]}]

# runs grid search using the above parameter and doing 5-fold cross validation
clf = GridSearchCV(SVC(C=1, gamma=0.01, class_weight='balanced',decision_function_shape='ovr'), tuned_parameters, cv=5, verbose=1)

# fit the data
clf.fit(xTrain, yTrain) 
```
After fitting the data, the class will use the best parameters from the k-fold cross validation and grid search for prediction. Prediction can now take place on the validation set and can get the classification accuracy, as well as F1, recall, and precision. 

```python
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# Predict Y labels for validation set
y_true, y_pred = yVal, clf.predict(xVal)

# get the accuracy score from prediction
accuracy = accuracy_score(y_true, y_pred)
print("Validation Accuracy: %.2f%%" % (accuracy * 100.0))

# print report containing f1, precision, and recall for both classes
print(classification_report(y_true, y_pred))
```

Aside from this simple example please explore the many modular scripts which provide a variety of ways to extra features and classify.

## Contributing

Please feel free to submit a push requests.

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

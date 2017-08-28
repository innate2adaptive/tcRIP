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

### Installing

A step by step series of examples that tell you have to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc

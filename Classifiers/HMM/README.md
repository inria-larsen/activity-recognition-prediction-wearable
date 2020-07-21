# Library for using HMM for activity recognition

The objective of this project is to adress the problem of recognizing the current activity performed by a human operator, for industrial applications related to automatic ergonomic evaluation. We propose a method based on wearable sensors and supervised learning based on Hidden Markov Model (HMM).

## Architecture

### Offline training

![alt text](https://github.com/inria-larsen/activity-recognition-prediction-wearable/blob/master/Classifiers/HMM/doc/img/diagram.png "Architecture offline")


## Installation

### Requirements

* The software was tested on windows and ubuntu 16
* python3.6

sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.6

* numpy: http://www.numpy.org/

sudo apt-get install python3-pip
pip3 install numpy

python3.6 -m pip install numpy
python3.6 -m pip install --upgrade pip

* hmmlearn : https://github.com/hmmlearn/hmmlearn

python3.6 -m pip install hmmlearn

* yarp: http://www.yarp.it/

not necessary to train models or visualize their output, but useful if you use these modules for online recognition. 


### Supported sensors

* MVN link suit from Xsens (MVN studio 4.4 used)
* e-glove from Emphasis Telematics SA




## Using the library offline

### Create a model

python3 model_simple.py -f [config_file] -c [configuration]

Config_file by default is config_model.ini. The "configuration" is the name of the model with its description in the config file.
There is a default model for tedstin, otherwise the most interesting models are GePos, DePos, CuAct, which correspond to the taxonomies of the paper Malaisé et al 2019 RA-L.
The data for training the model must be put into the path_data. The trained model will be saved in path_model.

Example of use:

python3 model_simple.py -f config_model.ini -c GePos

The output is two files: the first is a binary file with the content of the trained model; the second is a XML file with the list of possible outputs of the model, i.e., the labels of the states. The model, when used, outputs the probabilities for each state, and the label for each state is listed in the XML file.



### Compute automatically the best features for a model (feature selection)

python3 feature_selection.py -c [config_selection]

Config_selection is the set of parameters for the automatic feature selection process. The wrapper method is selected hardcoded in the text. The default config file is hardcoded, it is config_file_selection.ini. In the file you can choose if you want local of global features: local features are linked to a single limb or body part (e.g., pelvis position) whereas global features are processed quantities that consider the entire body (e.g., center of mass).
List_tracks is the list of the models that are used to find the features (features that are good for all of these listed models). Path_save is where you save the results. Method_sort is how you compute the global score of recognition for the different models: mean, hmean (harmonic mean), gmean (geometric mean).

Example of use:

python3 feature_selection.py -c GPos_only_local

The output is a .csv file with the results of feature selection: the features sorted in order of performance with their F1-score. Every line is a model with the corresponding features. Note that the files with 1,2,3 ... indicate how many features are in the model (1 feature, 2 features, etc.).

Note that feature selection can take a lot of time: on a single computer, exploring all the sets of possible features for up to 20 features for all the models GePos, CuAct, DePos and Det could take 3 weeks.



### Use a trained model to recognize an activity or posture

This is explained in online_recognition (other folder).



## Acknowledgments

The development of this software is partially supported by [the European Project H2020 An.Dy](http://andy-project.eu/).
We thank Emphasis Telematics SA (Dr. Giorgos Papapanagiotakis, Dr. Michalis Miatidis, Panos Stogiannos, Giannis Kantaris, Dimitris Potiriadis) for their support with the e-glove sensor.
e-glove is a registered trademark and property of Emphasis Telematics SA, providing the www.emphasisnet.gr/e-glove/#1476464465174-6fa88672-0410

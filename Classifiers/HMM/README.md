# Library for using HMM

The objective of this project is to adress the problem of recognizing the current activity performed by a human operator, for industrial applications related to automatic ergonomic evaluation. We propose a method based on wearable sensors and supervised learning based on Hidden Markov Model (HMM).

## Architecture

### Offline training

![alt text](https://github.com/inria-larsen/activity-recognition-prediction-wearable/blob/master/Classifiers/HMM/doc/img/diagram.png "Architecture offline")




## Supported sensors

* MVN link suit from Xsens (MVN studio 4.4 used)
* e-glove from Emphasis Telematics SA

## Requirement

* The software was tested on windows and ubuntu
* python3.6
* numpy: http://www.numpy.org/
* hmmlearn : https://github.com/hmmlearn/hmmlearn
* yarp: http://www.yarp.it/

## Compute the feature selection

python3 main.py --from [context_file]

## Create a model

python3 model_simple.py -f [config_file] -c [configuration]

## Acknowledgments

The development of this software is partially supported by [the European Project H2020 An.Dy](http://andy-project.eu/).
We thank Emphasis Telematics SA (Dr. Giorgos Papapanagiotakis, Dr. Michalis Miatidis, Panos Stogiannos, Giannis Kantaris, Dimitris Potiriadis) for their support with the e-glove sensor.
e-glove is a registered trademark and property of Emphasis Telematics SA, providing the www.emphasisnet.gr/e-glove/#1476464465174-6fa88672-0410

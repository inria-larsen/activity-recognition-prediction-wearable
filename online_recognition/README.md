# Online activity recognition


![alt text](https://github.com/inria-larsen/activity-recognition-prediction-wearable/blob/master/Classifiers/HMM/doc/img/diagram_online.png "Architecture online")

There are two main modules.

SensorProcessingModule (in sensor_processing.py) is used to retrieve the data from the Xsens suit. It process the data with a sliding window: the Xsens is at 240Hz, the window enables to lower the Hz and computes a mean of the signals over the window. It reads the data from a YARP port and outputs the result on another YARP port.

ActivityRecognitionModule (in activity_recognition.py) is the main module for online recognition. It reads the processed data from a YARP port connected to SensorProcessingModule, then outputs the result on two YARP ports: the first is /activity_recognition/probabilities:o, which contains the probabilities of every possible state (i.e., action label); the second is /activity_recognition/state:o is the most probable state, i.e., the action label with the highest probability.

The config files for these modules are YARP context files (to put in /build/share/yarp/context/online_recognition). 
For example, 





## Installation

### Supported sensors

* MVN link suit from Xsens (MVN studio 4.4 used)
* e-glove from Emphasis Telematics SA

### Requirements

* The software was tested on windows and ubuntu 16

* You need to have installed the class ModelHMM that is in Classifiers, that has the same requirements of these online reocgnition modules:

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

it is used to read data from the Xsens system through a yarp server. it is not used for training HMM models, so it is not necessary for visualization or local testing. it is however necessary for online use of the HMM models for action recognition using the Xsens system.
You must install it in your Ubuntu machine (where you run the activity recognition modules in python) and in the Windows machine where you have your Xsens software. Follow the yarp instructions to install it.

* Xsens streamer for Windows: https://gitlab.inria.fr/H2020-AnDy/sensors

this contains a mini server for streaming the Xsens data on a YARP port, as well a mini server for streaming the e-glove data on a YARP port.
Both are only available for Windows, and must run on the machine where the Xsens SDK is installed.


### Configure your Windows machine with the Xsens software

You must configure your Windows machine where you have your Xsens software installed. In our lab it is "AnDyExpe".

* Setting up yarp namespace : 

    Folder C:/Users/Researcher/AppData/Roaming/yarp/config/
    
    Create a file containing the IP address of the PC from which the yarp server is launched 
    
    For example: _demo_andy.conf :
    
      192.168.137.130 10000 yarp
      192.168.137.1 10000 yarp
      
    Change the configuration from a terminal: yarp namespace /demo_andy
    
  * Configure our Xsens streamer:
  
    Option > Preferences > Network Streamer :  
    
       add/enable configuration with HotSpot IP (192.168.137.1)
    
    Edit the streamer configuration file C:/ProgramData/yarp/xsens.ini :  
    
      IP_this_machine 192.168.137.1
      server_port_xsens 9763
    



## Running the demo 

### Prepare the Windows machine with Xsens and the Ubuntu machine with the activity recognition

On the Windows machine (AnDyExpe):

* Connect the Windows laptop to a network
* Open Network Settings -> Mobile Hotspot, and enable the HotSpot network. 
This is important because the Ubuntu laptop running the activity recognition modules must be connected to the Windows machine running the Xsens software, so that the demo can be executed independently of the wireless network, and both computers are on the same YARP network.


On the Ubuntu PC with the activity recognition module:

* Connect to the AnDyExpe HotSpot Network
* Start Yarp Server (yarp server [--write])


On the Windows machine (AnDyExpe):
* Launch MVN 2018 software
* Launch the Xsens streamer: Desktop/andy/sensors/xsens/yarp/build/Release/xsens.exe 


### Select your input: pre-recorded Xsens sequence or online stream from Xsens 

* with a pre-recorded Xsens sequence

You can run the demo with pre-recorded data from Xsens simply by charging a Xsens sequence. 
On your Windows machine where you have installed your Xsens MVN software: run the Xsens MVN software, open the file with your sequence, then click on "Play". You can also click on "Toggle repeat" so that the recording will loop. 

* connected to the Xsens MVN suit






## Xsens module: sensor_processing.py

python3 sensor_processing.py --from [context]

### Yarp port:

input: 
* /processing/xsens/"NameSignal":i

output:
* /processing/xsens/"NameSignal":o

## Activity recognition module

python3 activity_recognition.py --from [context]

### Yarp port:

input: 
* /activity_recognition/"NameSignal":i

output:
* /activity_recognition/state:o
* /activity_recognition/probabilities:o


## How to use it :

### Xsens

* Turn on the router/Access Point Xsens
* Equipping the suit
* Installing the Battery and BodyPack
* Turn on the BodyPack Xsens





### AnDyExpe


    
* Launch MVN 2018 software
    New session
    
    Wait for connection to the access point
    
    Perform calibration
    


### PC with activity recognition module

* Launch scripts :

        python3 sensor_processing --from [context_file]
    
        python3 activity_recognition.py --from [context_file]

### AnDyExpe
* Connect the streamer ports with sensor_processing :

        yarp connect /xsens/Signal /processing/xsens/signal:i

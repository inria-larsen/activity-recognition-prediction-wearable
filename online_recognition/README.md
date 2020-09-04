# Online activity recognition


![alt text](https://github.com/inria-larsen/activity-recognition-prediction-wearable/blob/master/Classifiers/HMM/doc/img/diagram_online.png "Architecture online")

There are two main modules.

SensorProcessingModule (in sensor_processing.py) is used to retrieve the data from the Xsens suit. It process the data with a sliding window: the Xsens is at 240Hz, the window enables to lower the Hz and computes a mean of the signals over the window. It reads the data from a YARP port and outputs the result on another YARP port.

ActivityRecognitionModule (in activity_recognition.py) is the main module for online recognition. It reads the processed data from a YARP port connected to SensorProcessingModule, then outputs the result on two YARP ports: the first is /activity_recognition/probabilities:o, which contains the probabilities of every possible state (i.e., action label); the second is /activity_recognition/state:o is the most probable state, i.e., the action label with the highest probability.

The config files for these modules are YARP context files (to put in /build/share/yarp/context/online_recognition). 






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
    
* Configure the Xsens MVN software to enable streaming :
  
   Open the Xsens MVN software and open in the tab Option > Preferences > Network Streamer. Check the table with IP and ports, and check the IP with your HotSpot IP.  
    
        add/enable configuration with HotSpot IP (192.168.137.1)
        
   In the list of datagram to stream, check that the following are selected:
        
        Position + Orientation (Quaternion)
        Linear Segment Kinematics
        Angular Segment Kinematics
        Joint Angles
        Center of Mass
    
 * Configure our Xsens streamer: 
 
 Edit the streamer configuration file C:/ProgramData/yarp/xsens.ini :  
    
        IP_this_machine 192.168.137.1. 
        server_port_xsens 9763
    


## Prepare the Windows machine with Xsens and the Ubuntu machine with the activity recognition

On the Windows machine (AnDyExpe):

* Connect the Windows laptop to a network
* Open Network Settings -> Mobile Hotspot, and enable the HotSpot network. 
This is important because the Ubuntu laptop running the activity recognition modules must be connected to the Windows machine running the Xsens software, so that the demo can be executed independently of the wireless network, and both computers are on the same YARP network.

On the Ubuntu PC with the activity recognition module:

* Connect to the AnDyExpe HotSpot Network 
* Start Yarp Server:

        yarp server [--write]

On the Windows machine (AnDyExpe):

* verify that you are on the same yarp network (should be, if you configured yarp correctly) by typing:
        
        yarp detect --write
    
* if you have problems, check that you are in the same yarp namespace:
        
        yarp namespace
        yarp where 


## Select your input: pre-recorded Xsens sequence or online stream from Xsens 

### Option A: with a pre-recorded Xsens sequence

You can run the demo with pre-recorded data from Xsens simply by charging a Xsens sequence. 
On your Windows machine where you have installed your Xsens MVN software (MVN Analyze 2018): run the Xsens MVN software, open the file with your sequence, then click on "Play". You can also click on "Toggle repeat" so that the recording will loop. 

On the Windows machine (AnDyExpe):

* Launch MVN 2018 software
* Open sequence file
* Click "Play"
* Click "Toggle repeat" to loop the recording
* Launch the Xsens streamer: 

        Desktop/andy/sensors/xsens/yarp/build/Release/xsens.exe 

### Option B: connected to the Xsens MVN suit

You need to start by preparing the Xsens suit.

* Turn on the router/Access Point of the Xsens suit.
* Wear the Xsens suit
* Install the Battery and BodyPack asking the help of a colleague.
* Turn on the BodyPack Xsens

On the Windows machine (AnDyExpe):

* Launch MVN 2018 software
* Start New session
* Wait for connection to the access point
* Perform calibration of the Xsens suit (with the walking calibration phase, just follow instructions)
* Launch the Xsens streamer: 

        Desktop/andy/sensors/xsens/yarp/build/Release/xsens.exe 

## Check before running the demo

You can now check that the yarp ports are streaming data. The list of available yarp port is:
        
        yarp name list

To read the content of a port do:
        
        yarp read ... <NAME_OF_THE_PORT>

For example:
        
        yarp read ... /xsens/JointAngles

## Run the demo

On the Ubuntu machine with the activity recognition module, launch the two scripts for the modules:

        python3 sensor_processing.py --from [context_file]
        python3 activity_recognition.py --from [context_file]

The context_file is the same for both modules. It is the YARP context folder where you have your configuration files. 
Example:

        python3 sensor_processing.py --from general_posture.ini
        python3 activity_recognition.py --from general_posture.ini
                

Check that you have all the YARP ports.
For the sensor processing module:
* input: 
        
        /processing/xsens/"NameSignal":i
* output: 
        
        /processing/xsens/"NameSignal":o

For the activity recognition module:
* input:
        
        /activity_recognition/"NameSignal":i

* output:
        
        /activity_recognition/state:o
        /activity_recognition/probabilities:o

Note that Activity Recognition is automatically connecting the ports of Sensor Processing at startup. So You must launch it imperatively after the other. 
If all the YARP ports are ok, then you can connect Sensor processing to the Xsens streamer and the demo will run automatically.
IMPORTANT: do the yarp connect from the Windows machine only!!

        yarp connect /xsens/Signal /processing/xsens/signal:i
                

    
## Visualization

You can check that the ports are sending the processed signals.
For example:

        yarp read ... /processing/xsens/Position/Pelvis_z:o
        yarp read ... /activity_recognition/state:o

To visualize the output of the demo, you can connect the YARP port of activity recognition to any GUI.
You can use those in https://github.com/inria-larsen/activity-recognition-prediction-wearable/tree/master/visualisation


    
# DEMOs with the different activity models

## Recognition with the general posture model   

It is only using 3 features. It outputs 4 main states (walking, standing, crouching, kneeling).

Check that the xsens streamer is streaming (on the Windows machine):

        yarp read ... /xsens/COM

Launch processing (on the Ubuntu machine):

        python3 sensor_processing.py --from demo_andy_final_general_posture.ini

Connect ports (from the Windows machine):

        yarp connect /xsens/LinearSegmentKinematics /processing/xsens/LinearSegmentKinematics:i
        <DO NOT CONNECT THE INIT COM PORT>
        
Check that the module is streaming (from the Ubuntu machine):

        yarp read ... /processing/xsens/Position/Pelvis_z:o

Launch activity recognition:

        python3 activity_recognition.py --from general_posture.ini
        
Check that the activity recognition module is working:

        yarp read ... /activity_recognition/state:o
        
## Recognition with the details model  

It is only using 6 features. It outputs 5 main states (overhead work, work above shoulder, upright, forward bent, strongly forward bent).

Check that the xsens streamer is streaming (on the Windows machine):

        yarp read ... /xsens/COM

Launch processing (on the Ubuntu machine):

        python3 sensor_processing.py --from demo_andy_final_details.ini

Connect ports (from the Windows machine):

        yarp connect /xsens/LinearSegmentKinematics /processing/xsens/LinearSegmentKinematics:i
        yarp connect /sens/AngularSegmentKinematics /processing/xsens/AngularSegmentKInematics:i
        <DO NOT CONNECT THE INIT COM PORT>
        
Check that the module is streaming (from the Ubuntu machine):

        yarp read ... /processing/xsens/Position/RightHand_x:o

Launch activity recognition:

        python3 activity_recognition.py --from demo_andy_final_details.ini
        
Check that the activity recognition module is working:

        yarp read ... /activity_recognition/state:o


        



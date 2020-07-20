# Online activity recognition


![alt text](https://github.com/inria-larsen/activity-recognition-prediction-wearable/blob/master/Classifiers/HMM/doc/img/diagram_online.png "Architecture online")

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

* Connecting to a network
* Launch HotSpot network

### PC with activity recognition module

* Connecting to the AnDyExpe HotSpot Network
Start Yarp Server (yarp server [--write])

### AnDyExpe

* Setting up yarp namespace : 

    Folder C:/Users/Researcher/AppData/Roaming/yarp/config/
    
    Create a file containing the IP address of the PC from which the yarp server is launched 
    
    For example: _demo_andy.conf :
    
      192.168.137.130 10000 yarp
      192.168.137.1 10000 yarp
      
    Change the configuration from a terminal: yarp namespace --config /demo_andy
    
* Launch MVN 2018 software
    New session
    
    Wait for connection to the access point
    
    Perform calibration
    
* Configure and launch Xsens streamer
    Preferences > Network Streamer : 
    
    add/enable configuration with HotSpot IP (192.168.137.1)
    
    Edit the streamer configuration file C:/ProgramData/yarp/xsens.ini :
    
      IP_this_machine 192.168.137.1
      server_port_xsens 9763
      
    Launch: Desktop/andy/sensors/xsens/yarp/build/Release/xsens.exe

### PC with activity recognition module

* Launch scripts :

        python3 sensor_processing --from [context_file]
    
        python3 activity_recognition.py --from [context_file]

### AnDyExpe
* Connect the streamer ports with sensor_processing :

        yarp connect /xsens/Signal /processing/xsens/signal:i

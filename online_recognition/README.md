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

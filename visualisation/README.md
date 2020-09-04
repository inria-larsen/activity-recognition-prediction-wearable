# Tools for visualization for the activity recognition demo

## Processing module

To visualize the signals of the processing module, do:

      python3 plot_signals.py --name_port <NAME_OF_THE_YARP_PORT> --size_window <N_DURATION>
      
This will open a plot for the signal streamed in the yarp port NAME_OF_THE_PORT, plotting only the last N_DURATION values.

Example:

      python3 plot_signals.py --name /processing/xsens/Position/Pelvis_z:o --size_window 3000
      

## Activity recognition: probabilities of states

To visualize the output of the activity_recognition module, do:

      python3 plot_probabilities.py
      
Without argument, it automatically connects to the ports of activity_recognition (so don't change names) and it shows the probabilities for each state.


## Activity recognition: only for detailed posture

There is a first version of a GUI for the detailed posture only. Go to src/qt and launch

      python3 reco_gui.py
      
It automatically connects to the ports of activity_recognition but only works with detailed posture. It shows a subset of possible states.

# Tools for visualizing the activity recognition

## Installation

### Requirements

## Run the scripts

To visualize the signals of the processing module, do:

      python3 plot_signals.py --name_port <NAME_OF_THE_YARP_PORT> --size_window <N_DURATION>
      
This will open a plot for the signal streamed in the yarp port NAME_OF_THE_PORT, plotting only the last N_DURATION values.

#!/bin/bash
python3 mainWithPythonLS5.py nb_epochs 2
for i in `seq 100 100 500`;
do
    python3 mainWithPythonLS5.py nb_epochs $i
done


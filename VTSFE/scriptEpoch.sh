#!/bin/bash

for i in `seq 10 10 500`;
do
    python3 mainWithPythonLS5.py nb_epochs $i
done


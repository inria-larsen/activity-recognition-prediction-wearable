# activity-recognition-prediction-wearable
Work in progress about activity recognition/prediction using wearable sensors information

# Variational Time Series Feature Extractor (VTSFE)

This is the implementation of an article proposed to [ICRA 2018](http://icra2018.org/).
Further computation details can be found in [Maxime Chaveroche's Master Thesis extract](./master_thesis_extract.pdf).

## Requirements

- Python 3
- Tensorflow
- numpy
- all other package indicated by python when running the main program

## Usage

`python3 main.py`

You can create new configuration files in the [app](./VTSFE/app) folder, instantiating classes from the subfolder [models](./VTSFE/app/models).
Then, you have to import them in [main.py](./VTSFE/main.py) and simply append a tuple to the *trainings* list in the form `(*model_name*, *config_instance*)`.

You can control the behaviour of the training loop through two booleans: *restore* and *train*.
The details of the *train* or *restore* behaviour could be set by commenting or uncommenting functions written there.

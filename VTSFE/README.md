# Variational Time Series Feature Extractor (VTSFE)

This is the implementation of an article proposed to [ICRA 2018](http://icra2018.org/). Configuration classes for VTSFE can be found in [tighter_lb_2D](./app/models/tighter_lb_2D.py) and [tighter_lb_light_2D](./app/models/tighter_lb_light_2D.py).
Further computation details can be found in [Maxime Chaveroche's Master Thesis extract](./master_thesis_excerpt.pdf).

VTSFE is inspired by Chen's [VAE-DMP](https://brml.org/uploads/tx_sibibtex/CheKarSma2016.pdf). A configuration class for our own implementation of VAE-DMP can be found in [vae_dmp_2D](./app/models/vae_dmp_2D.py).

VTSFE and VAE-DMP are based on Kingma's [VAE](https://arxiv.org/abs/1312.6114). A configuration class for our implementation of a chain of independent VAEs can be found in [vae_only_2D](./app/models/vae_only_2D.py).

## Requirements

- Python 3
- Tensorflow
- numpy
- all other package indicated by python when running the main program

## Usage

`python3 main.py`

You can create new configuration files in the [app](./app) folder, instantiating classes from the subfolder [models](./app/models).
Then, you have to import them in [main.py](./main.py) and simply append a tuple to the *trainings* list in the form `(model_name, config_instance)`.

You can control the behaviour of the training loop through two booleans: *restore* and *train*.
The details of the *train* or *restore* behaviour could be set by commenting or uncommenting functions written there.

# Variational Time Series Feature Extractor (VTSFE)

This is the implementation of an article proposed to [ICRA 2018](http://icra2018.org/). Configuration classes for VTSFE can be found in [tighter_lb_2D](./app/models/tighter_lb_2D.py) and [tighter_lb_light_2D](./app/models/tighter_lb_light_2D.py).
For the reviewers/readers of the VTSFE paper, we report in [Maxime Chaveroche's Master Thesis excerpt](./master_thesis_excerpt.pdf) the full computations leading to equations presented in Section 3 of our paper, precisely lower bound computations for Kingma's [VAE](https://arxiv.org/abs/1312.6114),  Karl's [DVBF](https://arxiv.org/pdf/1605.06432.pdf), Chen's [VAE-DMP](https://brml.org/uploads/tx_sibibtex/CheKarSma2016.pdf), and finally Chaveroche's VTSFE.

VTSFE is inspired by Chen's [VAE-DMP](https://brml.org/uploads/tx_sibibtex/CheKarSma2016.pdf). A configuration class for our own implementation of VAE-DMP can be found in [vae_dmp_2D](./app/models/vae_dmp_2D.py).

VTSFE and VAE-DMP are based on Kingma's [VAE](https://arxiv.org/abs/1312.6114). A configuration class for our implementation of a chain of independent VAEs can be found in [vae_only_2D](./app/models/vae_only_2D.py).

## Requirements

- Linux (not tested on Windows or Mac)
- Python 3
- [Tensorflow](https://www.tensorflow.org/install/) (Tested with Tensorflow 1.3)
- all other package indicated by python when running the main program
(For example, you can install them with `pip3 install package_name`)

## Usage

`python3 main.py`

You can create new configuration files in the [app](./app) folder, instantiating classes from the subfolder [models](./app/models).
Then, you have to import them in [main.py](./main.py) and simply append a tuple to the *trainings* list in the form `(model_name, config_instance)`.

You can control the behaviour of the training loop through two booleans: *restore* and *train*.
The details of the *train* or *restore* behaviour could be defined setting the boolean variables at the top of the loop.

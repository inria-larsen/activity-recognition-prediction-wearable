# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from vae import VAE


# Load MNIST data in a format suited for tensorflow.
# The script input_data is available under this URL:
# https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/tutorials/mnist/input_data.py
import input_data
mnist = input_data.read_data_sets('vae/MNIST_data', one_hot=True)


HYPERPARAMS = {
    "learning_rate": 1E-3,
    "activation_function": tf.nn.elu,
    "output_function": tf.nn.sigmoid,
    "vae_architecture": {
        "n_hidden_encoder_1": 500, # 1st layer encoder neurons
        "n_hidden_encoder_2": 500, # 2nd layer encoder neurons
        "n_hidden_decoder_1": 500, # 1st layer decoder neurons
        "n_hidden_decoder_2": 500, # 2nd layer decoder neurons
        "n_input": 784,            # dimensionality of observations (MNIST : 28x28)
        "n_z": 20                  # dimensionality of latent space
    }
}

TRAINING = {
    "data": mnist.train.images,
    "batch_size": 100,
    "nb_epochs": 10,
    "display_step": 2
}

# Training
vae = VAE(HYPERPARAMS, optimize=True)#.demo_2d()
vae.train(TRAINING)

x_sample = mnist.test.next_batch(100)[0]
x_reconstruct = vae.reconstruct(x_sample)


"""
    Examples of reconstruction for some MNIST observations
"""
plt.figure(figsize=(8, 12))
for i in range(5):
    plt.subplot(5, 2, 2*i + 1)
    plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
    plt.title("Test input")
    plt.colorbar()
    plt.subplot(5, 2, 2*i + 2)
    plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
    plt.title("Reconstruction")
    plt.colorbar()
plt.tight_layout()
plt.show()

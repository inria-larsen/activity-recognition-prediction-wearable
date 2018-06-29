# -*- coding: utf-8 -*-
import tensorflow as tf

from vae.vae import VAE
from models_dmp import DMP_NN
from models_dmp_vae import VAE_DMP


# data_source = "Nutan"
data_source = "MVNX"
data_path = "./xsens_data/xml"
data_types = ["position"]
plot_3D = True
unit_bounds = True

sub_sequences_size = 10
use_whole_sequence_for_forcing_term = True
continuity_penality = 0.5
use_z_derivative = False
transform_with_all_dvbf = True

DATA_PARAMS = {
    "nb_frames": 70,
    "nb_samples_per_mov": 10,
    "mov_types": [
        # AnDy mocaps
        "bent_fw",
        "bent_fw_strongly",
        "kicking",
        "lifting_box",
        "standing",
        "walking",
        "window_open",

        # CMU mocaps
        # "taichi",
        # "kicking",
        # "walking",
        # "punching"
    ]
}

TRAINING = {
    # # data shape = [nb_mov*nb_samples_per_mov, nb_frames, dim(values)] = [nb_samples, nb_frames, dim(values)]
    # "data_driver": data_driver,
    "batch_size": 1,
    "annealing_schedule_temp": None,
    "nb_epochs": 30,
    "display_step": 1,
    "checkpoint_step": 10
}

DMP_PARAMS = {
    "alpha": 2.,   # gain term (dimension homogeneous to inverse of time) applied on first order derivative
    "beta": 0.5,  # gain term (dimension homogeneous to inverse of time) applied on second order derivative
    "tau": 2.,     # time scaling term ----> integrate it into learned parameters since it varies with respect to movement speed
    "base_functions" : {
        "n": 50,    # if n == None, that model will build base functions every mu_step. Otherwise, mu_step is overwritten by a computed step based on n.
        "mu_step" : 2,
        "sigma": 2.5,
        "show": True #ori : j'ai change ici
    },
    "activation_function": tf.nn.elu,   # softmax. activation function used in article, but since it rescales input values to range [0, 1], all observations look the same => f is approximately the same for all movement types
    "output_function":     tf.identity,   # identity
    "f_architecture": {
        "h": 10,               # [5, 10]
    },
    "use_whole_sequence_for_forcing_term": use_whole_sequence_for_forcing_term,
    "model_continuity_error_coeff": continuity_penality,     # set to None if you don't want that model variable correction (system noise for DMP)
    "adapted_noise_inference": False,
    "model_monte_carlo_sampling": 30,
    "prior_monte_carlo_sampling": 5,
    "use_goal": False,
    "loss_on_dynamics": False
}

HYPERPARAMS = {
    "learning_rate": 1E-3,
    "activation_function": tf.nn.elu,
    # "output_function": tf.nn.sigmoid,
    "vae_architecture": {
        "L": 30,                   # number of Monte Carlo samples to reconstruct x (the observed value)
        "n_hidden_encoder_1": 200,  # 1st layer encoder neurons
        "n_hidden_encoder_2": None, # 2nd layer encoder neurons
        "n_hidden_decoder_1": 200,  # 1st layer decoder neurons
        "n_hidden_decoder_2": None, # 2nd layer decoder neurons
        # "n_input": len(data_driver.data[0][0]), # dimensionality of observations
        "n_z": 2,                    # dimensionality of latent space
        # "n_output": len(data_driver.data[0][0]) # dimensionality of observations
    },
    "delta": 0.5,
    "nb_frames": DATA_PARAMS["nb_frames"],
    "sub_sequences_size": sub_sequences_size,
    "sub_sequences_overlap": 5,
    "initial_std_vae_only_sequence_encoder": False,
    "input_sequence_size": 1,
    "target_sequence_offset": 0,
    "target_sequence_size": 1,    # if target_sequence_offset + target_sequence_size > input_sequence_size, target_sequence_size - input_sequence_size + offset indicates the predictive range
    # switch
    "std_vae_indices": [0, 1],
    # "std_vae_indices": range(sub_sequences_size),     # transforms dvbf in standard VAE through time (i.e. no transition, no model)
    "std_vae_class": VAE,
    "model_class": DMP_NN,
    "model_params": DMP_PARAMS,
    "model_vae_class": VAE_DMP,
    "optimize_each": False,
    "optimize_like_rnn": False,
    "separate_std_vaes": False,
    "bypass_back_prop_through_time": False,
    "block_decoder_when_training_transition": False,
    "block_encoder_when_training_std_vae": False,
    "unit_gaussian_cost_for_z": True,
    "absolute_last_frame": use_whole_sequence_for_forcing_term,
    "z_continuity_error_coeff": continuity_penality,     # set to None if you don't want that trajectory correction
    "use_z_derivative": use_z_derivative,
    "log_sigma_sq_values_limit": 20,
    "monte_carlo_recurrence": False
}

DATA_VISUALIZATION = {
    # "data_driver": data_driver,
    "nb_samples_per_mov": 1,
    # "displayed_movs": data_driver.mov_types,
    "plot_3D": plot_3D,
    "window_size": 15,
    "time_step": 5,
    "body_lines": True,
    "transform_with_all_dvbf": transform_with_all_dvbf
}

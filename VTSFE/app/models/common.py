# -*- coding: utf-8 -*-
import tensorflow as tf

from vae_standalone import VAE
from models_dmp import DMP_NN
from models_dmp_vae import VAE_DMP


class Common():
    """ Common configuration class """

    def __init__(self):

        self.VTSFE_PARAMS = {}
        self.DMP_PARAMS = {}
        self.TRAINING = {}
        self.DATA_PARAMS = {}
        self.DATA_VISUALIZATION = {}

        self.sub_sequences_size = 10
        self.continuity_penality = None

        self.TRAINING.update({
            "annealing_schedule_temp": None
        })

        self.DMP_PARAMS.update({
            "base_functions" : {
                "n": 50,    # if n == None, that model will build base functions every mu_step. Otherwise, mu_step is overwritten by a computed step based on n.
                "mu_step" : 2,
                "sigma": 2.5,
                "show": False
            },
            "activation_function": tf.nn.elu,
            "output_function":     tf.identity,   # identity
            "f_architecture": {
                "h": 10,               # [5, 10]
            },
            "use_whole_sequence_for_forcing_term": True,
            "model_continuity_error_coeff": self.continuity_penality,     # set to None if you don't want that model variable correction (system noise for DMP)
            "model_monte_carlo_sampling": 30,
            "scaling_noise": True,
            "use_dynamics_sigma": True,
            "only_dynamics_loss_on_mean": False,
            "forbidden_gradient_labels": ["gaussian_mixture", "system_noise_log_scale_sq"]
        })

        self.VTSFE_PARAMS.update({
            "learning_rate": 1E-3,
            "activation_function": tf.nn.elu,
            "output_function": tf.identity,
            "vae_architecture": {
                "L": 30,                   # number of Monte Carlo samples to reconstruct x (the observed value)
                "use_reconstr_sigma": False,
                "n_hidden_encoder_1": 200,  # 1st layer encoder neurons
                "n_hidden_encoder_2": None, # 2nd layer encoder neurons
                "n_hidden_decoder_1": 200,  # 1st layer decoder neurons
                "n_hidden_decoder_2": None, # 2nd layer decoder neurons
                "n_z": 2,                    # dimensionality of latent space
            },
            "delta": 0.5,
            "sub_sequences_size": self.sub_sequences_size,
            "sub_sequences_overlap": 8,
            "target_sequence_offset": 0,
            "target_sequence_size": 1,    # if target_sequence_offset + target_sequence_size > input_sequence_size, target_sequence_size - input_sequence_size + offset indicates the predictive range
            "std_vae_class": VAE,
            "model_class": DMP_NN,
            "model_params": self.DMP_PARAMS,
            "model_vae_class": VAE_DMP,
            "optimize_each": False,
            "optimize_like_rnn": False,
            "separate_std_vaes": False,
            "bypass_back_prop_through_time": False,
            "block_decoder_when_training_transition": False,
            "block_encoder_when_training_std_vae": False,
            "unit_gaussian_cost_for_z": True,
            "absolute_last_frame": self.DMP_PARAMS["use_whole_sequence_for_forcing_term"],
            "z_continuity_error_coeff": self.continuity_penality,     # set to None if you don't want that trajectory correction
            "log_sigma_sq_values_limit": 20,
            "separate_losses": False
        })


    def set_data_config(self, data_types, source):
        if data_types[0] == "position" and len(data_types) == 1:
            as_3D = True
        else:
            as_3D = False
        self.DATA_PARAMS.update({
            "data_source": source,
            "data_types": data_types,
            "as_3D": as_3D,
            "unit_bounds": True
        })

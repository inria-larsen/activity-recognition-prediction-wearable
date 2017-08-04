# -*- coding: utf-8 -*-
from .common import Common


class VAE_DMP_2D(Common):
    """ VAE_DMP_2D configuration class """

    def __init__(self):
        super(VAE_DMP_2D, self).__init__()

        self.DMP_PARAMS.update({
            "alpha": 2.,   # gain term (dimension homogeneous to inverse of time) applied on first order derivative
            "beta": 0.5,  # gain term (dimension homogeneous to inverse of time) applied on second order derivative
            "tau": 2.,
            "adapted_noise_inference": False,
            "use_goal": True,
            "loss_on_dynamics": False,
            "monte_carlo_recurrence": False
        })

        self.VTSFE_PARAMS.update({
            "initial_std_vae_only_sequence_encoder": True,
            "input_sequence_size": 5,
            "std_vae_indices": [0],
            "sub_sequences_overlap": 8,
            "use_z_derivative": True
        })

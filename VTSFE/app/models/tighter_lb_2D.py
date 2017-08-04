# -*- coding: utf-8 -*-
from .common import Common


class Tighter_LB_2D(Common):
    """ Tighter_LB_2D configuration class """

    def __init__(self):
        super(Tighter_LB_2D, self).__init__()

        self.DMP_PARAMS.update({
            "adapted_noise_inference": True,
            "prior_monte_carlo_sampling": 5,
            "use_goal": False,
            "loss_on_dynamics": True,
            "monte_carlo_recurrence": True
        })

        self.VTSFE_PARAMS.update({
            "initial_std_vae_only_sequence_encoder": False,
            "input_sequence_size": 1,
            "std_vae_indices": [0, 1],
            "sub_sequences_overlap": 8,
            "use_z_derivative": False
        })

# -*- coding: utf-8 -*-
from .vae_dmp_2D import VAE_DMP_2D


class VAE_DMP_No_Z_Derivative_2D(VAE_DMP_2D):
    """ VAE_DMP_2D without z derivative configuration class """

    def __init__(self):
        super(VAE_DMP_No_Z_Derivative_2D, self).__init__()

        self.VTSFE_PARAMS.update({
            "initial_std_vae_only_sequence_encoder": False,
            "input_sequence_size": 1,
            "std_vae_indices": [0, 1],
            "sub_sequences_overlap": 8,
            "use_z_derivative": False
        })

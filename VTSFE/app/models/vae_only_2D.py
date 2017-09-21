# -*- coding: utf-8 -*-
from .common import Common


class VAE_Only_2D(Common):
    """ VAE_Only_2D configuration class """

    def __init__(self):
        super(VAE_Only_2D, self).__init__()

        self.VTSFE_PARAMS.update({
            "input_sequence_size": 1,
            "std_vae_indices": range(self.sub_sequences_size),
            "use_z_derivative": False
        })

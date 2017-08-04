# -*- coding: utf-8 -*-
from .tighter_lb_2D import Tighter_LB_2D


class Tighter_LB_Light_2D(Tighter_LB_2D):
    """ Tighter_LB_Light_2D configuration class """

    def __init__(self):
        super(Tighter_LB_Light_2D, self).__init__()

        self.DMP_PARAMS.update({
            "prior_monte_carlo_sampling": 1
        })

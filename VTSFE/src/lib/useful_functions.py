# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib


def glorot_init(fan_in, fan_out, constant=1):
    """ Glorot initialization of network weights"""

    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))

    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


def inferior_limiter(x, lim):
    """ Limits the range of a function to values greater than or equal to lim
        limiter(x, lim) = relu(x - lim) + lim
                        = x         if x >= lim
                        = lim       otherwise
    """

    return tf.add( tf.nn.relu( tf.subtract(x, lim) ), lim)


def interval_limiter(x, lim):
    """ Limits the range of a function to values greater than or equal to lim
        limiter(x, lim) = -[relu(lim -[relu(x + lim) - lim]) - lim]
                        = lim       if x > lim
                        = -lim      if x < -lim
                        = x         otherwise
            where lim is a positive number
    """
    minus_one = tf.constant(-1, dtype=tf.float32)
    return tf.scalar_mul(
        minus_one,
        tf.subtract(
            tf.nn.relu(
                tf.subtract(
                    lim,
                    inferior_limiter(
                        x,
                        tf.scalar_mul(
                            minus_one,
                            lim
                        )
                    )
                )
            ),
            lim
        )
    )


def gaussian(t, mu, sigma):
    two_sigma_sq = 2*np.square(sigma)
    return np.divide(np.exp(-np.square(t - mu)/two_sigma_sq), np.sqrt(two_sigma_sq*np.pi))


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def get_available_cpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'CPU']


def get_available_devices():
    gpus = get_available_gpus()
    cpus = get_available_cpus()
    return gpus + cpus

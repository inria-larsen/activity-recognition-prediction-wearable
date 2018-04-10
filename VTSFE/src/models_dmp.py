# -*- coding: utf-8 -*-

from datetime import datetime
import pdb
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

from src.lib.useful_functions import *


class DMP_NN():
    """ Dynamic Movement Primitives model shaped as a Neural Network """

    PARAMS = {
        "alpha": 1.,   # gain term (dimension homogeneous to inverse of time) applied on first order derivative
        "beta": 0.25,  # gain term (dimension homogeneous to inverse of time) applied on second order derivative
        "tau": 1.,     # time scaling term ----> integrate it into learned parameters since it varies with respect to movement speed
        "base_functions" : {
            "n": 50,
            "mu_step" : 2,
            "variance": 2
        },
        "activation_function": tf.nn.elu,
        "output_function":     tf.nn.sigmoid,
        "f_architecture": {
            "h": 10,               # [5, 10]
        },
        "use_whole_sequence_for_forcing_term": True
    }


    def __init__(self, model_params, vtsfe):
        self.__dict__.update(self.PARAMS, **model_params)
        self.vtsfe = vtsfe
        if self.use_whole_sequence_for_forcing_term:
            self.nb_frames_mvt = self.vtsfe.nb_frames
            self.sequence_origin = 0
        else:
            self.nb_frames_mvt = self.vtsfe.sub_sequences_size
            self.sequence_origin = self.vtsfe.sub_sequence_origin

        self.init_base_functions(n=self.base_functions["n"], mu_step=self.base_functions["mu_step"])
        self.base_functions_samples = self.sample_base_functions(**self.base_functions)
        self.forcing_term_weights = self.forcing_term_weights_network(self.vtsfe.x_sequence[self.sequence_origin : self.sequence_origin + self.nb_frames_mvt])

        n_z = self.vtsfe.vae_architecture["n_z"]
        with tf.variable_scope("vae_dmp.encoder") as scope:
            # system_noise_log_scale_sq = tf.get_variable("system_noise_log_scale_sq", initializer=glorot_init(1, n_z))
            if self.scaling_noise:
                self.system_noise_log_scale_sq = tf.get_variable("system_noise_log_scale_sq", initializer=tf.zeros([1, n_z], dtype=tf.float32))
            else:
                self.system_noise_log_scale_sq = tf.add(
                    tf.zeros([1, n_z], dtype=tf.float32),
                    tf.constant(1, dtype=tf.float32)
                )
        print("system_noise_log_scale_sq weights initialized.")

        self.A = tf.constant([
                [   1.-self.alpha*self.beta*np.square(self.vtsfe.delta)/self.tau,     self.vtsfe.delta*(1.-self.alpha*self.vtsfe.delta/self.tau)],
                [   -self.alpha*self.beta*self.vtsfe.delta/self.tau,                  1.-self.alpha*self.vtsfe.delta/self.tau             ]
            ])

        if self.use_goal:
            # pre-compute values for transitions without z_derivative
            self.divider = tf.constant(1. + self.alpha*self.vtsfe.delta*self.tau/2., dtype=tf.float32)
            self.c_1 = tf.constant(2. - self.alpha*self.beta*self.tau*np.square(self.vtsfe.delta), dtype=tf.float32)
            self.c_2 = tf.constant(self.alpha*self.vtsfe.delta*self.tau/2. - 1., dtype=tf.float32)
            self.c_3 = tf.constant(self.tau*np.square(self.vtsfe.delta), dtype=tf.float32)
            self.c_4 = tf.constant(self.alpha*self.beta, dtype=tf.float32)
        else:
            self.c_3 = tf.constant(self.vtsfe.delta**2, dtype=tf.float32)


    def forcing_term_weights_network(self, x_sequence, reuse_weights=False):
        # x_sequence shape = [nb_frames, batch_size, n_input]

        self.network_weights = self.intialize_1hlayer_through_time(
            self.base_functions["n"],
            self.f_architecture["h"],
            self.nb_frames_mvt,
            self.vtsfe.vae_architecture["n_input"],
            self.vtsfe.vae_architecture["n_z"],
            reuse_weights=reuse_weights
        )

        weights = self.network_weights ["weights"]
        biases = self.network_weights ["biases"]
        hs = []

        nb_digits = len(str(self.nb_frames_mvt))
        for n in range(self.nb_frames_mvt):
            # hs shape = [nb_frames, batch_size, h]
            hs.append(
                self.activation_function(
                    tf.add(
                        tf.matmul(
                            x_sequence[n],
                            weights['h'+str(n).zfill(nb_digits)]
                        ),
                        biases['b'+str(n).zfill(nb_digits)]
                    )
                )
            )

        hs = tf.transpose(hs, perm=[1, 0, 2])
        # hs shape = [batch_size, h*nb_frames]
        hs = tf.reshape(hs, [-1, self.nb_frames_mvt*self.f_architecture["h"]])

        # infer the weights of the n base functions composing f
        # w_out shape = [batch_size, base_functions["n"]*n_z]
        w_out = self.output_function(
                    tf.add(
                        tf.matmul(
                            hs,
                            weights['w_out']
                        ),
                        biases['b_w_out']
                    )
                )

        # final shape = [n_z, batch_size, base_functions["n"]]
        return tf.transpose(
                    tf.reshape(
                        w_out,
                        [-1, self.base_functions["n"], self.vtsfe.vae_architecture["n_z"]]
                    ),
                    perm=[2, 0, 1]
                )


    def intialize_1hlayer_through_time(self, n_out, h, nb_frames, n_input, n_z, reuse_weights=False):
        all_weights = {}
        all_weights['weights'] = {}
        all_weights['biases'] = {}

        with tf.variable_scope("dmp.gaussian_mixture_weights", reuse=reuse_weights) as scope:
            nb_digits = len(str(nb_frames))
            for n in range(nb_frames):
                all_weights['weights']['h'+str(n).zfill(nb_digits)] = tf.get_variable('h'+str(n).zfill(nb_digits), initializer=glorot_init(n_input, h))
                all_weights['biases']['b'+str(n).zfill(nb_digits)] = tf.get_variable('b'+str(n).zfill(nb_digits), initializer=tf.zeros([h], dtype=tf.float32))

            all_weights['weights']['w_out'] = tf.get_variable('w_out', initializer=glorot_init(nb_frames*h, n_out*n_z))
            all_weights['biases']['b_w_out'] = tf.get_variable('b_w_out', initializer=tf.zeros([n_out*n_z], dtype=tf.float32))

        return all_weights


    def init_base_functions(self, n=None, mu_step=None):
        """ Sets self.nb_base_functions.
            Its value comes from either the user (if n isn't None), or is infered from nb_time_steps and mu_step
        """

        if n != None:
            self.nb_base_functions = n
        else:
            n_mod = self.nb_frames_mvt % mu_step
            if n_mod == 0:
                self.nb_base_functions = self.nb_frames_mvt/mu_step
            else:
                self.nb_base_functions = (self.nb_frames_mvt-n_mod)/mu_step + 1


    def sample_base_functions(self, n=None, mu_step=None, sigma=None, show=True):
        """ Generates T (number of time steps) samples from n gaussian functions with the same
        variance, each separated by mu_step, starting from t=0.

        Advised usage : - mu_step = 2 or 3 times the input sampling step
                        - variance = experimentally adjusted

        Output shape = [T, n]
        """

        if n != None:
            if n == 1:
                mu_step = 0
                mu = self.nb_frames_mvt / 2
            else:
                mu_step = self.nb_frames_mvt / (n-1)
                mu = 0

        if show:
            fig = plt.figure(0)
            fig.canvas.set_window_title('Gaussian Mixture - DMP Model')

        x = np.linspace(0, self.nb_frames_mvt, num=self.nb_frames_mvt)
        psis = []
        for i in range(self.nb_base_functions):
            values = []
            for t in x:
                values.append(gaussian(t, mu, sigma))
            psis.append(values)
            if show:
                plt.plot(x, values)
            mu += mu_step

        if show:
            plt.show()

        return tf.transpose(tf.constant(psis))


    def get_base_functions_samples(self, frame_number):
        f_n = tf.mod(
            tf.add(
                self.vtsfe.sub_sequence_origin,
                frame_number
            ),
            self.nb_frames_mvt
        )
        return self.base_functions_samples[f_n]


    def forcing_term_network(self, frame_number, weights=None):
        # weights shape = [n_z, batch_size, n]
        # base_functions_samples shape = [n]
        base_functions_samples = self.get_base_functions_samples(frame_number-1)

        if weights == None:
            f_weights = self.forcing_term_weights
        else:
            f_weights = weights

        # base_functions_samples shape = [n, 1]
        # -> useful for matmul since it only accepts 2D arguments
        base_functions_samples = tf.reshape(base_functions_samples, [-1, 1])
        f = []
        for i in range(self.vtsfe.vae_architecture["n_z"]):
            # w shape = [batch_size, n]
            w = f_weights[i]
            # f_component shape = [batch_size, 1]
            f_component = tf.matmul(w, base_functions_samples)
            # f shape = [n_z, batch_size]
            f.append(tf.reshape(f_component, [-1]))

        f = tf.divide(
            f,
            tf.reduce_sum(base_functions_samples)
        )

        # f shape = [batch_size, n_z]
        f = tf.transpose(f)

        # spatial scaling
        # f = tf.multiply(f, tf.subtract(self.vtsfe.goal_vae.z, self.vtsfe.origin.z))

        return f


    def model_wiring(self):
        # Only for lower bounds based on the two first priors
        if self.vtsfe.use_z_derivative:
            return
        if not self.loss_on_dynamics:
            return

        print("---- MODEL WIRING ----")

        if not self.only_dynamics_loss_on_mean and self.monte_carlo_recurrence:
            nb_recurrence_iter = self.prior_monte_carlo_sampling
        else:
            nb_recurrence_iter = 1
        nb_recurrence_iter_goal = 1
        # model_monte_carlo_sampling = 1
        # model_monte_carlo_sampling = self.model_monte_carlo_sampling

        if self.use_goal:
            nb_recurrence_iter_goal = nb_recurrence_iter

        x_reconstr_sequences = []
        # x_reconstr_sequences_log_sigma_sqs = []
        sys_noise_sequences_log_sigma_sqs = []
        forcing_terms = []
        for i, vae in enumerate(self.vtsfe.vae_subsequence):
            if i not in self.vtsfe.std_vae_indices:
                x_reconstr_sequences.append(vae.x_reconstr_means)
                # x_reconstr_sequences_log_sigma_sqs.append(vae.x_reconstr_log_sigma_sqs)
                sys_noise_sequences_log_sigma_sqs.append(vae.sys_noise_log_sigma_sqs)
                forcing_terms.append(vae.f)

        x_reconstr_sequences = np.array(x_reconstr_sequences)
        # x_reconstr_sequences_log_sigma_sqs = np.array(x_reconstr_sequences_log_sigma_sqs)
        sys_noise_sequences_log_sigma_sqs = np.array(sys_noise_sequences_log_sigma_sqs)
        forcing_terms = np.array(forcing_terms)

        if self.use_whole_sequence_for_forcing_term:
            x_reconstr_sequence_pre = self.vtsfe.x_sequence[:self.vtsfe.sub_sequence_origin]
            x_reconstr_sequence_post = self.vtsfe.x_sequence[self.vtsfe.sub_sequence_origin + self.vtsfe.sub_sequences_size : self.vtsfe.nb_frames]
        else:
            x_reconstr_sequence_pre = []
            x_reconstr_sequence_post = []

        f_costs = []
        for t in range(self.vtsfe.sub_sequences_size):
            f_costs.append([])

        n_input_log_2pi = tf.scalar_mul(
            tf.constant(self.vtsfe.vae_architecture["n_z"], dtype=tf.float32),
            tf.log(2. * np.pi)
        )
        half = tf.constant(0.5, dtype=tf.float32)
        log_inf = tf.constant(1E-05, dtype=tf.float32)

        for i in range(nb_recurrence_iter_goal):
            for j in range(nb_recurrence_iter):
                for k in range(nb_recurrence_iter):
                    device = self.vtsfe.devices[k % len(self.vtsfe.devices)]
                    with tf.device(device):
                        x_reconstr_sequence = x_reconstr_sequences[:, i, j, k, 0]
                        x_reconstr_sequence = np.concatenate([
                            [self.vtsfe.vae_subsequence[0].x_reconstr_means[j]],
                            [self.vtsfe.vae_subsequence[1].x_reconstr_means[k]],
                            x_reconstr_sequence
                        ], axis=0)

                        # conversion to tensor
                        x_seq = tf.concat([list(x_reconstr_sequence)], axis=0)
                        x_seq = tf.concat([x_reconstr_sequence_pre, x_seq, x_reconstr_sequence_post], axis=0)
                        # expensive computation
                        w = self.forcing_term_weights_network(x_seq, reuse_weights=True)
                        print("NEW FORCING TERM WEIGHTS "+str((i,j,k)))

                        for t in range(2, self.vtsfe.sub_sequences_size):
                            vae = self.vtsfe.vae_subsequence[t]
                            system_noise_log_scale_sq = tf.add(
                                vae.one_plus_t_minus_2_K_sq,
                                vae.system_noise_log_scale_sq
                            )
                            f = self.forcing_term_network(t, weights=w)

                            reconstr_loss_sq_sub = tf.square(tf.subtract(f, vae.f), name="f_reconstr_loss_sq_sub")
                            if self.use_dynamics_sigma:
                                reconstr_loss_division = tf.divide(reconstr_loss_sq_sub, tf.exp(system_noise_log_scale_sq), name="f_reconstr_loss_division")
                                # reconstr_losses shape = [batch_size, n_z]
                                reconstr_loss = tf.scalar_mul(
                                    half,
                                    tf.add(
                                        reconstr_loss_division,
                                        tf.add(
                                            system_noise_log_scale_sq,
                                            tf.log(2. * np.pi)
                                        )
                                    )
                                )
                                # reconstr_losses shape = [L**3, batch_size, n_z]
                                f_costs[t].append(reconstr_loss)
                            else:
                                f_costs[t].append(
                                    tf.scalar_mul(
                                        tf.constant(np.pi, dtype=tf.float32),
                                        reconstr_loss_sq_sub
                                    )
                                )

        for t in range(2, self.vtsfe.sub_sequences_size):
            vae = self.vtsfe.vae_subsequence[t]
            # f_cost[t] mean
            f_cost = tf.divide(
                tf.add_n(
                    f_costs[t]
                ),
                tf.constant(nb_recurrence_iter_goal*nb_recurrence_iter**2, dtype=tf.float32)
            )
            # insertion of that new cost in the corresponding vae
            vae.model_loss_raw = f_cost
            # vae.reconstr_loss = tf.add(vae.reconstr_loss, f_cost)
            vae.update_costs_and_variances()

        print("---- MODEL WIRING ---- END")


    def show_inferred_parameters(self, data_driver, x_samples, nb_samples_per_mov=1, displayed_movs=["window_open"]):
        """
            DMP's inferred parameters representation, i.e. :
                - forcing_term
                - system noise
        """
        labels = data_driver.data_labels
        nb_samples = len(x_samples)

        # don't display more movements than there are
        if nb_samples_per_mov > data_driver.nb_samples_per_mov:
            display_per_mov = data_driver.nb_samples_per_mov
        else:
            display_per_mov = nb_samples_per_mov

        plot_style = ["solid", "dashed"]
        inferred_p = ["f", "system noise"]
        nb_params = len(inferred_p)
        n_z = self.vtsfe.vae_architecture["n_z"]
        inferred_parameters = []

        i_p = ()
        for i, vae in enumerate(self.vtsfe.vae_subsequence):
            if i not in self.vtsfe.std_vae_indices:
                i_p += (vae.f, vae.sys_noise)

        # inferred_parameters shape = [nb_sub_sequences, nb_params*(sub_sequences_size-len(std_vae_indices)), nb_samples, n_z]
        inferred_parameters = self.vtsfe.get_values(i_p, x_samples, nb_variables_per_frame=nb_params, missing_value_indices=self.vtsfe.std_vae_indices)

        # list of sub sequences to plot
        inferred_parameters = np.array(inferred_parameters)

        # inferred_parameters final shape = [nb_sub_sequences, nb_frames, nb_params, nb_params, nb_samples, n_z]
        inferred_parameters = inferred_parameters.reshape([self.vtsfe.nb_sub_sequences, self.vtsfe.nb_frames, nb_params, nb_samples, n_z])

        colors = cm.rainbow(np.linspace(0, 1, len(displayed_movs)))

        def plot_sample(ax, s, i, k, flat_index, x, y, label=None):
            ax.plot(
                    range(self.vtsfe.nb_frames),
                    inferred_parameters[s, :, k, flat_index, x],
                    inferred_parameters[s, :, k, flat_index, y],
                    c=colors[i],
                    label=label,
                    linestyle=plot_style[k]
                )

        # turn on interactive mode
        plt.ion()

        # plot every permutation of 2 dimensions of latent space
        for x in range(self.vtsfe.vae_architecture["n_z"]):
            for y in range(x+1, self.vtsfe.vae_architecture["n_z"]):
                fig = plt.figure(figsize=(50,100))
                fig.canvas.set_window_title("DMP inferred_parameters")
                ax = fig.gca(projection='3d')
                ax.set_title("Axes = ["+str(x)+", "+str(y)+"]")

                # plot a path through per sample and per parameter, a color per movement type, a style per parameter type
                for i, mov in enumerate(displayed_movs):
                    for k in range(nb_params):
                        for j in range(display_per_mov):
                            flat_index = data_driver.mov_indices[mov] + j
                            for s in range(self.vtsfe.nb_sub_sequences):
                                # add only one instance of the same label
                                if j == 0 and s == 0:
                                    plot_sample(ax, s, i, k, flat_index, x, y, label=mov+" : "+inferred_p[k])
                                else:
                                    plot_sample(ax, s, i, k, flat_index, x, y)

                plt.legend(bbox_to_anchor=(0.8, 0.7), loc=2, borderaxespad=0.)
                plt.show()
                _ = input("Press [enter] to continue.") # wait for input from the user
                plt.close()    # close the figure to show the next one.

        # turn off interactive mode
        plt.ioff()

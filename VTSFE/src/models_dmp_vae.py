# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from lib.useful_functions import *
from vae_standalone import VAE



# Variational Autoencoder with Dynamic Movement Primitives as transition model
class VAE_DMP(VAE):

    MODEL_PARAMS = {
    }

    def __init__(self, hyperparams={}, model=None, vtsfe=None, z_continuity_error_coeff=None, model_continuity_error_coeff=None,
                    reuse_encoder_weights=False, reuse_decoder_weights=False, optimize=False,
                    input_tensor=None, target_tensor=None, frame_number=0, binary=False, save_path="vae_dmp-"+str(np.random.rand(1,1)[0]),
                    base_scope_encoder="vae_dmp", base_scope_decoder="vae", log_sigma_sq_values_limit=None):

        self.model = model
        self.vtsfe = vtsfe
        self.model_continuity_error_coeff = model_continuity_error_coeff

        self.parent_is_std_vae = False
        if frame_number-1 in self.vtsfe.std_vae_indices:
            self.parent_is_std_vae = True
        self.parent_vae = self.vtsfe.vae_subsequence[frame_number-1]

        if self.vtsfe.use_z_derivative:
            if frame_number == 1:
                self.parent_z_derivative = self.vtsfe.initial_z_derivative
            else:
                self.parent_z_derivative = self.parent_vae.z_derivative

            self.grand_parent_is_std_vae = self.parent_is_std_vae
            self.grand_parent = self.parent_vae
        else:
            self.grand_parent_is_std_vae = False
            if frame_number-2 in self.vtsfe.std_vae_indices:
                self.grand_parent_is_std_vae = True
            self.grand_parent = self.vtsfe.vae_subsequence[frame_number-2]

        self.frame_number = frame_number

        super(VAE_DMP, self).__init__(hyperparams, input_tensor=input_tensor, target_tensor=target_tensor, z_continuity_error_coeff=z_continuity_error_coeff,
                            save_path=save_path, reuse_encoder_weights=reuse_encoder_weights, optimize=optimize,
                            reuse_decoder_weights=reuse_decoder_weights, binary=binary, base_scope_encoder=base_scope_encoder, base_scope_decoder=base_scope_decoder,
                            log_sigma_sq_values_limit=log_sigma_sq_values_limit)


    def create_network(self):
        half = tf.constant(0.5, dtype=tf.float32)
        n_z = self.vae_architecture["n_z"]
        batch_size = tf.shape(self.x)[0]
        self.nb_recurrence_iter = 1
        self.nb_recurrence_iter_goal = 1

        if self.model.monte_carlo_recurrence:
            self.nb_recurrence_iter = self.model_params["prior_monte_carlo_sampling"]
            if self.model_params["use_goal"]:
                self.nb_recurrence_iter_goal = self.nb_recurrence_iter

        # Initialize autoencoder network weights and biases
        self.network_weights = self.initialize_weights()
        # forcing term f (DMP)
        self.f = self.model.forcing_term_network(self.frame_number)
        # system_noise_log_scale_sq shape = [batch_size, n_z]
        self.system_noise_log_scale_sq = tf.tile(self.model.system_noise_log_scale_sq, [batch_size, 1], name="system_noise_log_scale_sq_replication")
        self.system_noise_scale = tf.exp(
            tf.scalar_mul(
                half,
                self.system_noise_log_scale_sq
            )
        )

        # Use encoder network to determine mean and
        # (log) variance of Gaussian distribution in latent
        # space
        self.sys_noise_means = []
        self.sys_noise_unit_means = []
        self.sys_noise_log_sigma_sqs = []
        self.sys_noise_log_unit_sigma_sqs = []
        self.sys_noise_sigmas = []
        self.sys_noise_samples = []
        self.sys_noise_var_recurrence = []
        # if monte_carlo_recurrence, z_samples contains L^3 samples
        self.z_samples = []
        self.x_reconstr_means = []
        self.x_reconstr_log_sigma_sqs = []
        # monte carlo recurrence on goal prior
        for i in range(self.nb_recurrence_iter_goal):
            sys_noise_means_l2 = []
            sys_noise_unit_means_l2 = []
            sys_noise_log_sigma_sqs_l2 = []
            sys_noise_log_unit_sigma_sqs_l2 = []
            sys_noise_sigmas_l2 = []
            sys_noise_samples_l2 = []
            sys_noise_var_recurrence_l2 = []
            z_samples_l2 = []
            x_reconstr_means_l2 = []
            x_reconstr_log_sigma_sqs_l2 = []
            # first order monte carlo recurrence
            for j in range(self.nb_recurrence_iter):
                sys_noise_means_l3 = []
                sys_noise_unit_means_l3 = []
                sys_noise_log_sigma_sqs_l3 = []
                sys_noise_log_unit_sigma_sqs_l3 = []
                sys_noise_sigmas_l3 = []
                sys_noise_samples_l3 = []
                sys_noise_var_recurrence_l3 = []
                z_samples_l3 = []
                x_reconstr_means_l3 = []
                x_reconstr_log_sigma_sqs_l3 = []

                # second order monte carlo recurrence
                for k in range(self.nb_recurrence_iter):
                    device = self.vtsfe.devices[k % len(self.vtsfe.devices)]
                    with tf.device(device):
                        if self.parent_is_std_vae and self.grand_parent_is_std_vae:
                            z_grand_parent = self.grand_parent.z_samples[j]
                            z_parent = self.parent_vae.z_samples[k]
                        elif self.grand_parent_is_std_vae:
                            z_grand_parent = self.grand_parent.z_samples[k]
                            z_parent = self.parent_vae.z_samples[i,j,k,0]
                        else:
                            z_grand_parent = self.grand_parent.z_samples[i,j,k,0]
                            z_parent = self.parent_vae.z_samples[i,j,k,0]

                        if self.model_params["adapted_noise_inference"]:
                            model_noise_information = tf.concat([self.f, z_grand_parent, z_parent], 1)
                            if self.model_params["use_goal"]:
                                model_noise_information = tf.concat([self.vtsfe.goal_vae.z[i], model_noise_information], 1)
                        else:
                            model_noise_information = z_parent

                        encoded_inputs = self.encoder_network(
                            model_noise_information,
                            self.network_weights["weights_encoder"],
                            self.network_weights["biases_encoder"]
                        )

                        if self.model.monte_carlo_recurrence:
                            if self.grand_parent_is_std_vae:
                                sys_noise_var_recurrence = tf.constant(0, dtype=tf.float32)
                                if self.parent_is_std_vae:
                                    sys_noise_var_parent = tf.constant(0, dtype=tf.float32)
                                else:
                                    sys_noise_var_parent = tf.exp(self.parent_vae.sys_noise_log_unit_sigma_sqs[i,j,k])
                            else:
                                sys_noise_var_recurrence = self.parent_vae.sys_noise_var_recurrence[i,j,k]
                                sys_noise_var_parent = tf.exp(self.parent_vae.sys_noise_log_unit_sigma_sqs[i,j,k])

                            sys_noise_var_recurrence = tf.add(
                                sys_noise_var_recurrence,
                                tf.scalar_mul(
                                    tf.square(self.model.c_3),
                                    sys_noise_var_parent
                                )
                            )
                            sys_noise_var_recurrence_l3.append(sys_noise_var_recurrence)
                        else:
                            sys_noise_var_recurrence = tf.constant(0, dtype=tf.float32)

                        sys_noise_var = tf.log(
                            tf.add(
                                tf.exp(encoded_inputs[1]),
                                sys_noise_var_recurrence
                            )
                        )

                        sys_noise_mean = encoded_inputs[0]

                        sys_noise_means_l3.append(
                            tf.multiply(
                                self.system_noise_scale,
                                sys_noise_mean
                            )
                        )
                        sys_noise_unit_means_l3.append(
                            sys_noise_mean
                        )
                        sys_noise_log_sigma_sqs_l3.append(
                            tf.add(
                                self.system_noise_log_scale_sq,
                                sys_noise_var
                            )
                        )
                        sys_noise_log_unit_sigma_sqs_l3.append(
                            sys_noise_var
                        )
                        sys_noise_sigmas_l3.append(
                            tf.exp(
                                tf.scalar_mul(
                                    half,
                                    sys_noise_log_sigma_sqs_l3[k]
                                )
                            )
                        )
                        # Monte Carlo sampling
                        # sys_noise_sample = mu + sigma*epsilon
                        # Draw L samples of system noise from Gaussian distribution
                        # sys_noise_samples shape = [L, batch_size, n_z]
                        sys_noise_samples_l4 = [sys_noise_means_l3[k]]
                        for l in range(self.model_params["model_monte_carlo_sampling"]-1):
                            eps = tf.random_normal([batch_size, n_z], mean=0.0, stddev=1.0, dtype=tf.float32)
                            sys_noise_samples_l4.append(
                                tf.add(
                                    sys_noise_means_l3[k],
                                    tf.multiply(
                                        sys_noise_sigmas_l3[k],
                                        eps
                                    )
                                )
                            )
                        sys_noise_samples_l3.append(sys_noise_samples_l4)

                        if self.vtsfe.use_z_derivative:
                            # z_samples, z_derivative_samples shape = [L, batch_size, n_z]
                            z_samples_l4, self.z_derivative_samples = self.make_transition_with_z_derivative(
                                sys_noise_samples_l4,
                                self.vtsfe.goal_vae.z,
                                self.parent_vae.z,
                                self.parent_z_derivative
                            )
                            self.z_derivative_samples = np.array([[[self.z_derivative_samples]]])
                            z_samples_l3.append(z_samples_l4)
                            # to allow direct retrieving of z derivative
                            # self.z_derivative = tf.reduce_mean(self.z_derivative_samples, 0)
                            x_reconstr_means_l4, x_reconstr_log_sigma_sqs_l4 = self.decoder_network(
                                z_samples_l4,
                                self.network_weights["weights_decoder"],
                                self.network_weights["biases_decoder"]
                            )
                            x_reconstr_means_l3.append(x_reconstr_means_l4)
                            x_reconstr_log_sigma_sqs_l3.append(x_reconstr_log_sigma_sqs_l4)
                        else:
                            if self.model_params["use_goal"]:
                                # z_samples shape = [L, batch_size, n_z]
                                z_samples_l4 = self.make_transition_without_z_derivative(
                                    sys_noise_samples_l4,
                                    self.vtsfe.goal_vae.z_samples[i],
                                    z_parent,
                                    z_grand_parent
                                )
                            else:
                                # z_samples shape = [L, batch_size, n_z]
                                z_samples_l4 = self.make_transition_without_attractor(
                                    sys_noise_samples_l4,
                                    z_parent,
                                    z_grand_parent
                                )

                            z_samples_l3.append(z_samples_l4)

                            x_reconstr_means_l4, x_reconstr_log_sigma_sqs_l4 = self.decoder_network(
                                z_samples_l4,
                                self.network_weights["weights_decoder"],
                                self.network_weights["biases_decoder"]
                            )
                            x_reconstr_means_l3.append(x_reconstr_means_l4)
                            x_reconstr_log_sigma_sqs_l3.append(x_reconstr_log_sigma_sqs_l4)

                sys_noise_means_l2.append(sys_noise_means_l3)
                sys_noise_unit_means_l2.append(sys_noise_unit_means_l3)
                sys_noise_log_sigma_sqs_l2.append(sys_noise_log_sigma_sqs_l3)
                sys_noise_log_unit_sigma_sqs_l2.append(sys_noise_log_unit_sigma_sqs_l3)
                sys_noise_sigmas_l2.append(sys_noise_sigmas_l3)
                sys_noise_samples_l2.append(sys_noise_samples_l3)
                sys_noise_var_recurrence_l2.append(sys_noise_var_recurrence_l3)
                z_samples_l2.append(z_samples_l3)
                x_reconstr_means_l2.append(x_reconstr_means_l3)
                x_reconstr_log_sigma_sqs_l2.append(x_reconstr_log_sigma_sqs_l3)

            self.sys_noise_means.append(sys_noise_means_l2)
            self.sys_noise_unit_means.append(sys_noise_unit_means_l2)
            self.sys_noise_log_sigma_sqs.append(sys_noise_log_sigma_sqs_l2)
            self.sys_noise_log_unit_sigma_sqs.append(sys_noise_log_unit_sigma_sqs_l2)
            self.sys_noise_sigmas.append(sys_noise_sigmas_l2)
            self.sys_noise_samples.append(sys_noise_samples_l2)
            self.sys_noise_var_recurrence.append(sys_noise_var_recurrence_l2)
            self.z_samples.append(z_samples_l2)
            self.x_reconstr_means.append(x_reconstr_means_l2)
            self.x_reconstr_log_sigma_sqs.append(x_reconstr_log_sigma_sqs_l2)

        self.sys_noise_means = np.array(self.sys_noise_means)
        self.sys_noise_unit_means = np.array(self.sys_noise_unit_means)
        self.sys_noise_log_sigma_sqs = np.array(self.sys_noise_log_sigma_sqs)
        self.sys_noise_log_unit_sigma_sqs = np.array(self.sys_noise_log_unit_sigma_sqs)
        self.sys_noise_sigmas = np.array(self.sys_noise_sigmas)
        self.sys_noise_samples = np.array(self.sys_noise_samples)
        self.sys_noise_var_recurrence = np.array(self.sys_noise_var_recurrence)
        self.z_samples = np.array(self.z_samples)
        self.x_reconstr_means = np.array(self.x_reconstr_means)
        self.x_reconstr_log_sigma_sqs = np.array(self.x_reconstr_log_sigma_sqs)

        # to allow direct retrievings
        self.sys_noise = self.sys_noise_samples[0,0,0,0]

        # deterministic chaining:
        # takes z_t based on the couple of means of (z_(t-1), z_(t-2))
        self.z = self.z_samples[0,0,0,0]
        if self.vtsfe.use_z_derivative:
            self.z_derivative = self.z_derivative_samples[0,0,0,0]
        # self.sys_noise_log_sigma_sq = tf.reduce_mean(sys_noise_log_sigma_sqs, 0)
        # in order to make model continuity correction agnostic
        self.var_to_correct = self.sys_noise
        # to allow direct retrieving of reconstructed input x
        # x_reconstr shape = [batch_size, n_output]
        self.x_reconstr = self.x_reconstr_means[0,0,0,0]


    def make_transition_without_attractor(
        self,
        sys_noises,
        z,
        previous_z
    ):
        next_z = []
        # compute values for each Monte Carlo sample
        for i in range(len(sys_noises)):
            # next_z shape = [L, batch_size, n_z]
            next_z.append(
                tf.subtract(
                    tf.add(
                        tf.scalar_mul(
                            self.model.c_3,
                            tf.add(
                                self.f,
                                sys_noises[i]
                            )
                        ),
                        tf.scalar_mul(
                            tf.constant(2, dtype=tf.float32),
                            z
                        )
                    ),
                    previous_z
                )
            )
        return next_z


    def make_transition_without_z_derivative(
        self,
        sys_noises,
        z_goal,
        z,
        previous_z
    ):
        """ Transition model without z derivative
        """

        m = self.model
        next_z = []
        without_noise_part = tf.subtract(
            tf.add(
                tf.scalar_mul(
                    m.c_1,
                    z
                ),
                tf.scalar_mul(
                    m.c_2,
                    previous_z
                )
            ),
            tf.scalar_mul(
                m.c_3,
                tf.add(
                    tf.scalar_mul(
                        m.c_4,
                        z_goal
                    ),
                    self.f
                )
            )
        )

        # compute values for each Monte Carlo sample
        for i in range(len(sys_noises)):
            # next_z shape = [L, batch_size, n_z]
            next_z.append(
                tf.divide(
                    tf.subtract(
                        without_noise_part,
                        tf.scalar_mul(
                            m.c_3,
                            sys_noises[i]
                        )
                    ),
                    m.divider
                )
            )

        return next_z


    def make_transition_with_z_derivative(
        self,
        sys_noises,
        z_goal,
        z,
        z_derivative
    ):
        """ Transition model with z derivative (as in CheKarSma2016)
        """

        m = self.model

        # A shape = [2, 2]
        A = m.A
        next_z = []
        next_z_derivative = []
        # pre-compute values regardless of the latent dimension or Monte Carlo sample
        b_sum_zg_f = tf.add(
            tf.scalar_mul(
                tf.constant(m.alpha*m.beta, dtype=tf.float32),
                z_goal
            ),
            self.f
        )

        b_sq_deltau = tf.constant(np.square(self.vtsfe.delta)/m.tau, dtype=tf.float32)
        b_deltau = tf.constant(self.vtsfe.delta/m.tau, dtype=tf.float32)

        bs = []
        # pre-compute values for each Monte Carlo sample regardless of the latent dimension
        for i in range(len(sys_noises)):
            b_sum = tf.add(
                b_sum_zg_f,
                sys_noises[i]
            )
            # bs = [L, 2, batch_size, n_z]
            bs.append(
                [
                    tf.scalar_mul( b_sq_deltau, b_sum ),
                    tf.scalar_mul( b_deltau ,   b_sum )
                ]
            )

        # compute the z and z_derivative transitions for each latent dimension and for each Monte Carlo sample
        for i in range(self.vae_architecture["n_z"]):
            # slice keeps all batch samples but only keeps the ith dimension
            # zs shape = [2, batch_size, 1]
            zs = [
                tf.slice(z, [0, i], [-1, 1]),
                tf.slice(z_derivative, [0, i], [-1, 1])
            ]
            # zs shape = [2, batch_size]
            zs = tf.reshape(zs, [2, -1])

            z_samples = []
            z_derivative_samples = []
            for j in range(len(sys_noises)):
                # b shape = [2, batch_size, 1]
                b = tf.slice(bs[j], [0, 0, i], [-1, -1, 1])
                # b shape = [2, batch_size]
                b = tf.reshape(b, [2, -1])

                # Prediction
                next_zs = tf.add(tf.matmul(A, zs), b)
                # z_samples shape = [L, batch_size]
                z_samples.append(next_zs[0])
                # z_derivative_samples shape = [L, batch_size]
                z_derivative_samples.append(next_zs[1])

            # next_z shape = [n_z, L, batch_size]
            next_z.append(z_samples)
            # next_z_derivative shape = [n_z, L, batch_size]
            next_z_derivative.append(z_derivative_samples)

        # next_z shape = [L, batch_size, n_z]
        next_z = tf.transpose(next_z, perm=[1, 2, 0])
        # next_z_derivative shape = [L, batch_size, n_z]
        next_z_derivative = tf.transpose(next_z_derivative, perm=[1, 2, 0])

        output_z = []
        output_z_derivative = []
        for l in range(len(sys_noises)):
            output_z.append(next_z[l])
            output_z_derivative.append(next_z_derivative[l])

        return output_z, output_z_derivative


    def initialize_encoder(self, all_weights):
        n_z = self.vae_architecture["n_z"]
        if self.model_params["adapted_noise_inference"]:
            dim_factor = 3
            if self.model_params["use_goal"]:
                dim_factor = 4
        else:
            dim_factor = 1
        n_additional = self.vae_architecture["n_z"]*dim_factor

        weights = self.initialize_2hlayers(self.base_scope_encoder, "encoder", self.reuse_encoder_weights, all_weights,
            self.vae_architecture["n_hidden_encoder_1"],
            self.vae_architecture["n_hidden_encoder_2"],
            self.n_input+n_additional,
            n_z
        )

        return weights


    def encoder_network(self, model_noise_information, weights, biases):
        # Generate probabilistic encoder (encoder network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.activation_function(tf.add(tf.matmul(tf.concat([self.x, model_noise_information], 1), weights['h1']),
                                           biases['b1']))

        if 'h2' in weights:
            hidden_layer = self.activation_function(tf.add(tf.matmul(layer_1, weights['h2']),
                                               biases['b2']))
        else:
            hidden_layer = layer_1

        sys_noise_mean = tf.add(tf.matmul(hidden_layer, weights['out_mean']),
                        biases['out_mean'])
        sys_noise_log_sigma_sq = tf.add(tf.matmul(hidden_layer, weights['out_log_sigma']),
                   biases['out_log_sigma'])

        n_z = self.vae_architecture["n_z"]

        if self.log_sigma_sq_values_limit != None:
            # Limiting values of log_sigma_sq to keep numerical stability
            # = keeping sigma >= exp(-limit/2) and <= exp(limit/2)
            sys_noise_log_sigma_sq = interval_limiter(sys_noise_log_sigma_sq, self.log_sigma_sq_values_limit)

        return (sys_noise_mean, sys_noise_log_sigma_sq)


    def create_latent_loss(self):
        # annealing schedule to give f priority on system noise during training
        c_ta = self.vtsfe.annealing_schedule

        # Kullback-Leibler divergence between two multivariate gaussians
        # with identity variance for the second dstribution
        # and eventually annealing schedule

        one = tf.constant(1, dtype=tf.float32)
        half = tf.constant(0.5, dtype=tf.float32)
        # n_z = tf.constant(self.vae_architecture["n_z"], dtype=tf.float32)
        latent_loss_addn = []

        if self.model.monte_carlo_recurrence:
            nb_samples = self.nb_recurrence_iter_goal*self.nb_recurrence_iter**2
            frame_number = self.frame_number
        else:
            nb_samples = 1
            frame_number = 2

        K_sq = tf.square(self.model.c_3)
        self.one_plus_t_minus_2_K_sq = tf.add(
            one,
            tf.scalar_mul(
                K_sq,
                tf.constant(
                    self.frame_number-2,
                    dtype=tf.float32
                )
            )
        )

        for i in range(self.nb_recurrence_iter_goal):
            for j in range(self.nb_recurrence_iter):
                for k in range(self.nb_recurrence_iter):
                    latent_loss_addn.append(
                        tf.scalar_mul(
                            half,
                            tf.add(
                                # (c_ta-1)*ln(2*pi)
                                tf.scalar_mul(
                                    tf.log(2*np.pi),
                                    tf.subtract(
                                        c_ta,
                                        one
                                    )
                                ),
                                tf.subtract(
                                    # c_ta*(ln(1+K^2(t-2)) + [mean^2 + sigma^2]/(1+K^2(t-2)))
                                    tf.scalar_mul(
                                        c_ta,
                                        tf.add(
                                            tf.log(self.one_plus_t_minus_2_K_sq),
                                            tf.divide(
                                                tf.add(
                                                    tf.square(
                                                        self.sys_noise_unit_means[i,j,k]
                                                    ),
                                                    tf.exp(
                                                        self.sys_noise_log_unit_sigma_sqs[i,j,k]
                                                    )
                                                ),
                                                self.one_plus_t_minus_2_K_sq
                                            )
                                        )
                                    ),
                                    # 1 + ln(sigma^2)
                                    tf.add(
                                        one,
                                        self.sys_noise_log_unit_sigma_sqs[i,j,k]
                                    )
                                )
                            )
                        )
                    )

        # a_schedule_log_2pi = tf.scalar_mul(
        #     tf.constant(0.5*self.vae_architecture["n_z"]*np.log(2*np.pi), dtype=tf.float32),
        #     tf.subtract(
        #         c_ta,
        #         tf.constant(1, dtype=tf.float32)
        #     )
        # )
        # latent_loss shape = [batch_size, n_z]
        latent_loss = tf.divide(
            tf.add_n(latent_loss_addn),
            tf.constant(nb_samples, dtype=tf.float32)
        )
        return latent_loss


    def create_loss_function(self):
        """ The loss is composed of two terms:
        1.) log p(x|z) => reconstruction loss
          - log-likelihood of x given z, sampled L times then averaged (Monte Carlo)
          - in binary case: cross-entropy
          - in continuous case: log-likelihood of seeing the target x under the
            Gaussian distribution parameterized by x_reconstr_mean, sigma = sqrt(exp(x_reconstr_log_sigma_sq))

        2.) The latent loss, which is defined as the Kullback Leibler divergence
            between the system noise distribution induced by the encoder on
            the data and a unit gaussian prior. This acts as a kind of regularizer.

        N.B.: The variational lower bound represented by these two previously defined losses must be maximized. But, as tensorflow
        minimizes the network cost function, we mutiply that lower bound by -1 and minimize it.
        """

        reconstr_losses = []
        for l1 in range(self.nb_recurrence_iter_goal):
            for l2 in range(self.nb_recurrence_iter):
                for l3 in range(self.nb_recurrence_iter):
                    reconstr_losses.append(
                        self.create_reconstruction_loss(
                            self.x_reconstr_means[l1][l2][l3],
                            self.x_reconstr_log_sigma_sqs[l1][l2][l3],
                            self.model_params["model_monte_carlo_sampling"]*self.nb_recurrence_iter_goal*self.nb_recurrence_iter**2
                        )
                    )
        self.reconstr_loss_raw = tf.add_n(reconstr_losses)
        self.model_loss_raw = tf.zeros([tf.shape(self.x)[0], self.vae_architecture["n_z"]])
        self.latent_loss_raw = self.create_latent_loss()

        # annealing schedule to give f priority on system noise during training
        c_ta = self.vtsfe.annealing_schedule
        self.reconstr_loss_raw = tf.scalar_mul(c_ta, self.reconstr_loss_raw)
        self.continuity_loss_per_dim = tf.zeros(self.z.get_shape()[1])
        self.update_costs_and_variances()


    def compute_global_cost(self):
        self.reconstr_loss = tf.reduce_sum(
            self.reconstr_loss_raw,
            1
        )
        self.latent_loss = tf.reduce_sum(
            self.latent_loss_raw,
            1
        )
        self.model_loss = tf.reduce_sum(
            self.model_loss_raw,
            1
        )
        self.cost_add = tf.add_n([self.reconstr_loss, self.latent_loss, self.model_loss])
        self.add_continuity_corrections()


    def update_costs_and_variances(self):
        super(VAE_DMP, self).update_costs_and_variances()

        self.model_loss_per_dim = tf.reduce_mean(
            self.model_loss_raw,
            0
        )
        # batch averages and variances
        self.model_loss_avg, self.model_loss_var = tf.nn.moments(self.model_loss, [0])


    def add_continuity_corrections(self):
        self.continuity_loss_per_dim = tf.zeros(self.z.get_shape()[1])
        if self.z_continuity_error_coeff is not None:
            self.add_z_correction_cost()
            if self.vtsfe.use_z_derivative:
                self.add_z_derivative_correction_cost()

        if self.model_continuity_error_coeff is not None:
            self.add_sys_noise_correction_cost()

        self.cost_add = tf.add(
            self.cost_add,
            tf.reduce_sum(
                self.continuity_loss_per_dim
            )
        )


    def add_sys_noise_correction_cost(self):
        self.var_target = tf.placeholder(tf.float32, [None, self.vae_architecture["n_z"]], name="sys_noise_target")
        self.var_correction = tf.square(
            tf.subtract(
                self.var_to_correct,
                self.var_target
            )
        )
        self.var_correction = tf.scalar_mul(
            tf.constant(self.model_continuity_error_coeff, dtype=tf.float32),
            tf.reduce_mean(
                self.var_correction,
                0
            )
        )
        self.continuity_loss_per_dim = tf.add(
            self.continuity_loss_per_dim,
            self.var_correction
        )

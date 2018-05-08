# -*- coding: utf-8 -*-

from datetime import datetime
import os.path
import pdb

import numpy as np
import tensorflow as tf

from src.lib.useful_functions import *
from .vtsfe_plots import VTSFE_Plots
from .my_statistics import My_statistics


# Variational Time Series Feature Extractor
class VTSFE():

    VTSFE_PARAMS = {
        "learning_rate": 1E-3,
        "activation_function": tf.nn.softplus,
        "output_function": tf.nn.sigmoid,
        "vae_architecture": {
            "L": 100,                   # number of Monte Carlo samples to reconstruct x (the observed value)
            "n_hidden_encoder_1": 500, # 1st layer encoder neurons
            "n_hidden_encoder_2": 500, # 2nd layer encoder neurons
            "n_hidden_decoder_1": 500, # 1st layer decoder neurons
            "n_hidden_decoder_2": 500, # 2nd layer decoder neurons
            "n_input": 50,              # dimensionality of observations
            "n_z": 5                    # dimensionality of latent space
        },
        "delta": 1.
    }

    def __init__(
        self,
        vtsfe_params={},
        save_path=None,
        restore=False,
        training_mode=True
    ):

        self.__dict__.update(self.VTSFE_PARAMS, **vtsfe_params)
        self.save_path = save_path
        self.restore = restore
        self.training_mode = training_mode

        # if self.std_vae_indices:
        #     if 0 not in self.std_vae_indices:
        #         if len(self.std_vae_indices) == self.sub_sequences_size:
        #             self.std_vae_indices[0] = 0
        #         else:
        #             self.std_vae_indices = [0] + self.std_vae_indices
        # else:
        #     self.std_vae_indices = [0]
        self.std_vae_indices = self.std_vae_indices[:self.sub_sequences_size]

        self.log_sigma_sq_values_limit = tf.constant(self.log_sigma_sq_values_limit, dtype=tf.float32)

        self.sub_sequences_step = self.sub_sequences_size - self.sub_sequences_overlap
        self.nb_sub_sequences = int((self.nb_frames - self.sub_sequences_size) / self.sub_sequences_step) + 1
        self.sub_sequence_origin = tf.placeholder(tf.int32, shape=[], name="sub_sequence_origin")

        self.x = []
        for n in range(self.sub_sequences_size):
            self.x.append(tf.placeholder(tf.float32, shape=[None, self.vae_architecture["n_input"]], name="x"+str(n)))

        self.x = np.array(self.x)

        self.x_sequence = tf.placeholder(tf.float32, shape=[self.nb_frames, None, self.vae_architecture["n_input"]], name="x_sequence")

        self.annealing_schedule = tf.placeholder(tf.float32, shape=[], name="annealing_schedule")

        self.network_variables = []

        self.vae_only = True
        if len(self.std_vae_indices) < self.sub_sequences_size:
            self.vae_only = False
            # if there is model VAEs in the VTSFE network
            self.model = self.model_class(self.model_params, self)

        if self.input_sequence_size == 1:
            self.initial_std_vae_only_sequence_encoder = False

        if self.input_sequence_size > 1 and not self.initial_std_vae_only_sequence_encoder:
            self.std_vae_sequence_encoders = True
        else:
            self.std_vae_sequence_encoders = False

        self.devices = get_available_gpus()
        if not self.devices:
            self.devices = get_available_cpus()

        # Create a chain of vaes
        self.create_network()
        if not self.vae_only:
            # if there is model VAEs in the VTSFE network
            # Extra-wiring specific to model
            self.model.model_wiring()

        if not self.initial_std_vae_only_sequence_encoder and not self.std_vae_sequence_encoders and self.z_continuity_error_coeff is not None and self.use_z_derivative:
            self.update_initial_derivative_continuity_loss()

        self.get_network_variables()

        # Get each VAE loss function and variance computation
        self.get_costs_and_variances()

        if not self.optimize_each:
            # Defines loss function based on all VAEs in network
            self.create_loss_function()

        # Defines the optimizer to use
        self.create_loss_optimizer()

        # create VTSFE session
        self.session = tf.Session()

        # create a summary writer, add the 'graph' to the event file.
        self.writer = tf.summary.FileWriter("./log_dir", self.session.graph)

        # init session to restore or randomly initialize all variables in network
        self.init_session()

        self.vtsfe_plots = VTSFE_Plots(self)

        # self.writer.flush()
        # self.writer.close()


    def create_network(self):
        def concatenate_sequence(sequence, indices, offset=0, input_is_tensor=True):
            seq = []
            if input_is_tensor:
                for k in indices:
                    # seq shape = [input_sequence_size, batch_size, n_input]
                    seq.append(
                        sequence[tf.add(offset, k)]
                    )
            else:
                for k in indices:
                    # offset must be an integer here
                    # seq shape = [input_sequence_size, batch_size, n_input]
                    seq.append(sequence[offset+k])
            # ouput shape = [batch_size, input_sequence_size*n_input]
            return tf.concat(seq, 1)

        print("\nVTSFE NETWORK ---- INITIATE ORIGINAL STANDARD VAE :")

        if self.absolute_last_frame:
            #ori : ??
            # if we want an absolute last frame reference, then you have to create an additional std VAE for an absolute first frame reference
            edge_vae_data_tensor = self.x_sequence
        else:
            # otherwise, just create the first std VAE of subsequence
            edge_vae_data_tensor = self.x

        if self.std_vae_sequence_encoders or self.initial_std_vae_only_sequence_encoder:
            #ori : ??
            # in the case of goal vae, use the same decoder as other VAEs to build the same latent space
            # but encode information differently from them since you want to encode information to reconstruct the last frame contained in input sequence,
            # instead of encoding the first frame from that sequence like the other std VAEs do
            reuse_std_encoder_weights_for_goal = False
            base_scope_goal_encoder = "goal_vae"
        else:
            # in the case of goal vae, use the same decoder as other VAEs to build the same latent space
            # and encode information the same way the other std VAEs do since you simply want to reconstruct the whole input
            reuse_std_encoder_weights_for_goal = True
            base_scope_goal_encoder = "origin_vae"

        estimate_z_derivative = False
        if self.std_vae_sequence_encoders or self.initial_std_vae_only_sequence_encoder:
            origin_input_tensor = concatenate_sequence(
                edge_vae_data_tensor,
                range(self.input_sequence_size)
            )
            if self.use_z_derivative:
                estimate_z_derivative = True
        else:
            origin_input_tensor = edge_vae_data_tensor[0]

        self.origin = self.std_vae_class(
            self.__dict__,
            input_tensor=origin_input_tensor,
            target_tensor=edge_vae_data_tensor[0],
            reuse_encoder_weights=False,
            reuse_decoder_weights=False,
            optimize=False,
            base_scope_encoder="origin_vae",
            estimate_z_derivative=estimate_z_derivative,
            z_continuity_error_coeff=self.z_continuity_error_coeff,
            apply_latent_loss=self.unit_gaussian_cost_for_z,
            log_sigma_sq_values_limit=self.log_sigma_sq_values_limit
        )

        if not self.sub_sequences_size-1 in self.std_vae_indices:
            print("\nVTSFE NETWORK ---- INITIATE GOAL STANDARD VAE :")

            if self.std_vae_sequence_encoders:
                goal_input_tensor = concatenate_sequence(
                    edge_vae_data_tensor,
                    range(-self.input_sequence_size, 0)
                )
            else:
                goal_input_tensor = edge_vae_data_tensor[-1]

            self.goal_vae = self.std_vae_class(
                self.__dict__,
                input_tensor=goal_input_tensor,
                target_tensor=edge_vae_data_tensor[-1],
                reuse_encoder_weights=reuse_std_encoder_weights_for_goal,
                reuse_decoder_weights=True,
                optimize=False,
                base_scope_encoder=base_scope_goal_encoder,
                z_continuity_error_coeff=self.z_continuity_error_coeff,
                apply_latent_loss=self.unit_gaussian_cost_for_z,
                log_sigma_sq_values_limit=self.log_sigma_sq_values_limit
            )

        print("\nVTSFE NETWORK ---- INITIATE "+str(len(self.std_vae_indices))+" STANDARD VAEs :")

        self.vae_subsequence = []
        self.std_vaes_like_origin = []

        if self.std_vae_indices and self.input_sequence_size + self.std_vae_indices[-1] > self.sub_sequences_size:
            std_vae_data_tensor = self.x_sequence
            offset = self.sub_sequence_origin
            input_is_tensor = True
        else:
            std_vae_data_tensor = self.x
            offset = 0
            input_is_tensor = False

        base_scope_encoder="origin_vae"
        reuse_encoder_weights = True
        #crée toutes les VAEs
        for i in range(self.sub_sequences_size):
            if i in self.std_vae_indices:
                print("---------- "+str(i))

                # if we take relative edge values and if we want to encode z directly every subsequence
                # then self.origin and self.vae_subsequence[0] are the same VAE
                if i == 0 and not self.absolute_last_frame:
                    self.vae_subsequence.append(
                        self.origin
                    )
                else:
                    if self.z_continuity_error_coeff is not None and i < self.sub_sequences_overlap:
                        z_continuity_error_coeff = self.z_continuity_error_coeff
                    else:
                        z_continuity_error_coeff = None

                    estimate_z_derivative = False
                    if self.std_vae_sequence_encoders or (self.initial_std_vae_only_sequence_encoder and i == 0):
                        input_tensor = concatenate_sequence(
                            std_vae_data_tensor,
                            range(i, i + self.input_sequence_size),
                            offset=offset,
                            input_is_tensor=input_is_tensor
                        )
                        if self.use_z_derivative:
                            estimate_z_derivative = True
                    else:
                        input_tensor = self.x[i]

                    vae = self.std_vae_class(
                        self.__dict__,
                        input_tensor=input_tensor,
                        target_tensor=self.x[i],
                        reuse_encoder_weights=reuse_encoder_weights,
                        reuse_decoder_weights=True,
                        optimize=False,
                        base_scope_encoder=base_scope_encoder,
                        estimate_z_derivative=estimate_z_derivative,
                        z_continuity_error_coeff=z_continuity_error_coeff,
                        apply_latent_loss=self.unit_gaussian_cost_for_z,
                        log_sigma_sq_values_limit=self.log_sigma_sq_values_limit
                    )

                    self.std_vaes_like_origin.append(
                        vae
                    )
                    self.vae_subsequence.append(
                        vae
                    )
                    if self.initial_std_vae_only_sequence_encoder and i == 0:
                        reuse_encoder_weights = False
                        base_scope_encoder = "vae"
                    if i == 1:
                        reuse_encoder_weights = True
            else:  # ????
                self.vae_subsequence.append(None)

        if self.use_z_derivative:
            self.estimate_initial_z_derivative()

        print("\nVTSFE NETWORK ---- INITIATE "+str(self.sub_sequences_size - len(self.std_vae_indices))+" MODEL VAEs :")

        transition_initiated = False
        for i in range(self.sub_sequences_size):
            if i not in self.std_vae_indices: #complete les none d'avant?
                print("---------- "+str(i))

                #revient à : le premier ne reutilise pas les poids de l'encodeur
                if transition_initiated:
                    reuse_encoder_weights = True
                else:
                    reuse_encoder_weights = False
                    transition_initiated = True

                z_continuity_error_coeff = None
                model_continuity_error_coeff = None
                if i < self.sub_sequences_overlap:
                    z_continuity_error_coeff = self.z_continuity_error_coeff
                    model_continuity_error_coeff = self.model_params["model_continuity_error_coeff"]

                self.vae_subsequence[i] = self.model_vae_class(
                    self.__dict__,
                    model=self.model,
                    vtsfe=self,
                    input_tensor=self.x[i],
                    target_tensor=self.x[i],
                    frame_number=i,
                    z_continuity_error_coeff=z_continuity_error_coeff,
                    model_continuity_error_coeff=model_continuity_error_coeff,
                    reuse_encoder_weights=reuse_encoder_weights,
                    reuse_decoder_weights=True,
                    optimize=False,
                    log_sigma_sq_values_limit=self.log_sigma_sq_values_limit
                )

        self.vae_subsequence = np.array(self.vae_subsequence)

        print("\nVTSFE NETWORK ---- END")


    def estimate_initial_z_derivative(self):
        self.initial_z_derivative_variables = []

        if self.initial_std_vae_only_sequence_encoder or self.std_vae_sequence_encoders:
            z_derivative = self.vae_subsequence[0].z_derivative

            if self.z_continuity_error_coeff is not None:
                self.initial_z_derivative_target = self.vae_subsequence[0].z_derivative_target
                self.initial_z_derivative_correction = self.vae_subsequence[0].z_derivative_correction
        else:
            print("\nZ DERIVATIVE ESTIMATION NETWORK ---- INITIATE "+str(self.n_estimate_deriv-1)+" ADDITIONAL STANDARD VAEs :")

            vaes = [ self.vae_subsequence[0] ]
            zs = []
            for i in range(1, self.input_sequence_size):
                print("---------- "+str(i))
                vaes.append(
                    self.std_vae_class(
                        self.__dict__,
                        input_tensor=self.x[i],
                        reuse_encoder_weights=True,
                        reuse_decoder_weights=True,
                        optimize=False,
                        log_sigma_sq_values_limit=self.log_sigma_sq_values_limit
                    )
                )

            for i in range(self.input_sequence_size):
                # zs shape = [input_sequence_size, batch_size, n_z]
                zs.append(vaes[i].z)

            n_z = self.vae_architecture["n_z"]
            # zs shape = [batch_size, input_sequence_size, n_z]
            zs = tf.transpose(zs, perm=[1, 0, 2])
            # zs shape = [batch_size, input_sequence_size*n_z]
            zs = tf.reshape(zs, [-1, self.n_estimate_deriv*n_z])

            # Fully connected layer to compute z derivative (since we don't have information about frame -1, we can't directly compute the derivative.)
            weights = tf.get_variable("z_derivative_estimation_weights", initializer=glorot_init(self.input_sequence_size*n_z, n_z))
            biases = tf.get_variable("z_derivative_estimation_biases", initializer=tf.zeros([n_z]), dtype=tf.float32)

            self.initial_z_derivative_variables.append(weights)
            self.initial_z_derivative_variables.append(biases)

            self.network_variables += self.initial_z_derivative_variables

            print(weights.name+" initialized.")
            print(biases.name+" initialized.")

            # z_derivative shape = [batch_size, n_z]
            z_derivative = tf.add(
                                tf.matmul(
                                    zs,
                                    weights
                                ),
                                biases
                            )

            if self.z_continuity_error_coeff is not None:
                self.initial_z_derivative_target = tf.placeholder(tf.float32, shape=[None, self.vae_architecture["n_z"]], name="initial_z_derivative_target")
                self.initial_z_derivative_correction = tf.square(
                    tf.subtract(
                        z_derivative,
                        self.initial_z_derivative_target
                    )
                )
                self.initial_z_derivative_correction = tf.scalar_mul(
                    tf.constant(self.z_continuity_error_coeff, dtype=tf.float32),
                    self.initial_z_derivative_correction
                )

            print("Z DERIVATIVE ESTIMATION NETWORK ---- END\n")

        self.initial_z_derivative = z_derivative
        if self.z_continuity_error_coeff is not None:
            self.reduced_initial_z_derivative_correction = tf.reduce_sum(
                self.initial_z_derivative_correction,
                1
            )
            self.initial_z_derivative_cost, self.initial_z_derivative_variance = tf.nn.moments(self.reduced_initial_z_derivative_correction, [0], name="initial_z_derivative_cost_and_variance")


    def update_initial_derivative_continuity_loss(self):
        self.vae_subsequence[0].continuity_loss_per_dim = tf.add(
            self.vae_subsequence[0].continuity_loss_per_dim,
            self.initial_z_derivative_correction
        )
        self.vae_subsequence[0].continuity_loss_avg = tf.reduce_mean(self.vae_subsequence[0].continuity_loss)
        self.vae_subsequence[0].cost_add = tf.add(
            self.vae_subsequence[0].cost_add,
            self.vae_subsequence[0].continuity_loss
        )
        self.vae_subsequence[0].cost, self.vae_subsequence[0].variance = tf.nn.moments(self.vae_subsequence[0].cost_add, [0])


    def get_costs_and_variances(self):
        self.cost_per_vae = []
        self.var_per_vae = []
        self.reconstr_cost_per_vae = []
        self.reconstr_var_per_vae = []
        self.latent_cost_per_vae = []
        self.latent_var_per_vae = []
        self.continuity_cost_per_vae = []
        self.model_cost_per_vae = []
        self.model_var_per_vae = []
        self.reconstr_cost_per_dim = []
        self.latent_cost_per_dim = []
        self.continuity_cost_per_dim = []
        self.model_cost_per_dim = []
        self.decoder_cost_per_dim = []
        self.encoder_cost_per_dim = []

        def get_cost_and_variance(vae, is_std):
            m, v = vae.cost, vae.variance
            reco_m, reco_v = vae.reconstr_loss_avg, vae.reconstr_loss_var
            lat_m, lat_v = vae.latent_loss_avg, vae.latent_loss_var
            reco_d = vae.reconstr_loss_per_dim
            lat_d = vae.latent_loss_per_dim
            if is_std:
                zero = tf.constant(0, tf.float32)
                model_m, model_v = zero, zero
                model_d = tf.zeros([self.vae_architecture["n_z"]])
            else:
                model_m, model_v = vae.model_loss_avg, vae.model_loss_var
                model_d = vae.model_loss_per_dim
            # cost_per_vae shape = [sub_sequences_size]
            # var_per_vae shape = [sub_sequences_size]
            self.cost_per_vae.append(m)
            self.var_per_vae.append(v)
            self.reconstr_cost_per_vae.append(reco_m)
            self.reconstr_var_per_vae.append(reco_v)
            self.latent_cost_per_vae.append(lat_m)
            self.latent_var_per_vae.append(lat_v)
            self.model_cost_per_vae.append(model_m)
            self.model_var_per_vae.append(model_v)
            self.reconstr_cost_per_dim.append(reco_d)
            self.latent_cost_per_dim.append(lat_d)
            self.model_cost_per_dim.append(model_d)
            self.decoder_cost_per_dim.append(
                reco_d
            )
            self.encoder_cost_per_dim.append(
                tf.add(
                    model_d,
                    lat_d
                )
            )
            if self.z_continuity_error_coeff is not None:
                cont_m = vae.continuity_loss_avg
                self.continuity_cost_per_vae.append(cont_m)
                cont_d = vae.continuity_loss_per_dim
                self.continuity_cost_per_dim.append(cont_d)

        get_cost_and_variance(
            self.origin,
            True
        )

        for vae in self.std_vaes_like_origin:
            get_cost_and_variance(
                vae,
                True
            )

        for i, vae in enumerate(self.vae_subsequence):
            if i not in self.std_vae_indices:
                get_cost_and_variance(
                    vae,
                    False
                )

        if not self.sub_sequences_size-1 in self.std_vae_indices:
            get_cost_and_variance(
                self.goal_vae,
                True
            )


    def create_loss_function(self):
        """
        Creates a global loss function for VTSFE based on each VAE loss function
        """
        # costs = []
        #
        # def get_cost(vae, costs):
        #     costs.append(vae.cost_add)
        #
        # if not self.separate_std_vaes:
        #     get_cost(
        #         self.origin,
        #         costs
        #     )
        #
        #     for vae in self.std_vaes_like_origin:
        #         get_cost(
        #             vae,
        #             costs
        #         )
        #
        #     if not self.sub_sequences_size-1 in self.std_vae_indices:
        #         get_cost(
        #             self.goal_vae,
        #             costs
        #         )
        #
        #     if self.z_continuity_error_coeff is not None and self.use_z_derivative:
        #         costs.append(self.reduced_initial_z_derivative_correction)
        #
        # for i, vae in enumerate(self.vae_subsequence):
        #     if i not in self.std_vae_indices:
        #         get_cost(vae, costs)
        #
        # # sum on all frames
        # cost = tf.add_n(costs)
        # # average over batch
        # self.cost = tf.reduce_mean(cost, 0)
        # # self.costs = tf.reduce_mean(costs, 1)

        # sum on all frames
        self.reconstr_cost = tf.add_n(self.reconstr_cost_per_vae)
        self.latent_cost = tf.add_n(self.latent_cost_per_vae)

        if not self.vae_only:
            self.model_cost = tf.add_n(self.model_cost_per_vae)

        self.cost = tf.reduce_mean(self.cost_per_vae, 0)
        self.cost_per_dim_and_frame = tf.concat([
            self.reconstr_cost_per_dim,
            self.latent_cost_per_dim
        ], axis=1)

        if not self.vae_only:
            self.cost_per_dim_and_frame = tf.concat([
                self.cost_per_dim_and_frame,
                self.model_cost_per_dim
            ], axis=1)

        if self.z_continuity_error_coeff is not None:
            self.continuity_cost = tf.add_n(self.continuity_cost_per_vae)
            self.cost_per_dim_and_frame = tf.concat([
                self.cost_per_dim_and_frame,
                self.continuity_cost_per_dim
            ], axis=1)
            self.continuity_cost_per_dim_and_frame = tf.identity(
                self.continuity_cost_per_dim
            )

        self.decoder_cost_per_dim_and_frame = tf.identity(
            self.reconstr_cost_per_dim
        )
        if not self.vae_only:
            self.model_cost_per_dim_and_frame = tf.identity(
                self.model_cost_per_dim
            )
        self.encoder_cost_per_dim_and_frame = tf.identity(
            self.latent_cost_per_dim
        )

    def create_loss_optimizer(self):
        # Use ADAM optimizer
        self.optimizer = tf.train.AdamOptimizer(
            name="Adam",
            learning_rate=self.learning_rate
        )
        if self.training_mode:
            self.compute_gradients()


    def get_network_variables(self):
        def get_sorted_values_from_dict(dic, values, index=slice(0, None)):
            sorted_values = values
            def getKey(item):
                return item[0][index]

            sorted_items = sorted(dic.items(), key=getKey)
            for _, value in sorted_items:
                sorted_values.append(value)
            return sorted_values

        def delete_duplicates_of_origin(var_list):
            # delete possible duplicates of origin VAE variables from var_list
            for variable in (self.origin_vae_encoder_variables+self.origin_vae_decoder_variables):
                if variable in var_list:
                    del var_list[var_list.index(variable)]

        model_variables = {}
        if len(self.std_vae_indices) < self.sub_sequences_size:
            model_variables = self.model.network_weights
        origin_vae_variables = self.origin.network_weights
        if not self.sub_sequences_size-1 in self.std_vae_indices:
            goal_vae_variables = self.goal_vae.network_weights
        model_vae_variables = {}

        for i, vae in enumerate(self.vae_subsequence):
            if i not in self.std_vae_indices:
                model_vae_variables = vae.network_weights
                break

        # get origin VAE encoder variables
#[<tf.Variable 'origin_vae.encoder/h1:0' shape=(66, 200) dtype=float32_ref>,
# <tf.Variable 'origin_vae.encoder/out_log_sigma_weights:0' shape=(200, 2) dtype=float32_ref>,
# <tf.Variable 'origin_vae.encoder/out_mean_weights:0' shape=(200, 2) dtype=float32_ref>
        self.origin_vae_encoder_variables = get_sorted_values_from_dict(origin_vae_variables["weights_encoder"], [])
#<tf.Variable 'origin_vae.encoder/b1:0' shape=(200,) dtype=float32_ref>, 
#<tf.Variable 'origin_vae.encoder/out_log_sigma_biases:0' shape=(2,) dtype=float32_ref>,
# <tf.Variable 'origin_vae.encoder/out_mean_biases:0' shape=(2,) dtype=float32_ref>]
        self.origin_vae_encoder_variables = get_sorted_values_from_dict(origin_vae_variables["biases_encoder"], self.origin_vae_encoder_variables)

        # get origin VAE decoder variables
#[<tf.Variable 'vae.decoder/h1:0' shape=(2, 200) dtype=float32_ref>, 
#<tf.Variable 'vae.decoder/out_mean_weights:0' shape=(200, 66) dtype=float32_ref>, 
#<tf.Variable 'vae.decoder/b1:0' shape=(200,) dtype=float32_ref>, 
#<tf.Variable 'vae.decoder/out_mean_biases:0' shape=(66,) dtype=float32_ref>]

        self.origin_vae_decoder_variables = get_sorted_values_from_dict(origin_vae_variables["weights_decoder"], [])
        self.origin_vae_decoder_variables = get_sorted_values_from_dict(origin_vae_variables["biases_decoder"], self.origin_vae_decoder_variables)

        if not self.sub_sequences_size-1 in self.std_vae_indices:
            # get goal VAE encoder variables (might be different from the origin VAE ones in case of usage of sequence encoders)
            self.goal_vae_encoder_variables = get_sorted_values_from_dict(goal_vae_variables["weights_encoder"], [])
            self.goal_vae_encoder_variables = get_sorted_values_from_dict(goal_vae_variables["biases_encoder"], self.goal_vae_encoder_variables)

        # get all remaining variables in network
#[<tf.Variable 'dmp.gaussian_mixture_weights/h00:0' shape=(66, 10) dtype=float32_ref>,
#...
#<tf.Variable 'dmp.gaussian_mixture_weights/h69:0' shape=(66, 10) dtype=float32_ref>, 
#<tf.Variable 'dmp.gaussian_mixture_weights/w_out:0' shape=(700, 100) dtype=float32_ref>]
        for dic in model_variables.values():
            self.network_variables = get_sorted_values_from_dict(dic, self.network_variables)
#<tf.Variable 'vae_dmp.encoder/b1:0' shape=(200,) dtype=float32_ref> ...
        for dic in model_vae_variables.values():
            self.network_variables = get_sorted_values_from_dict(dic, self.network_variables)

        # delete possible duplicates of origin VAE variables from network_variables
        delete_duplicates_of_origin(self.network_variables)

        if not self.sub_sequences_size-1 in self.std_vae_indices:
            # delete possible duplicates of origin VAE variables from goal_vae_variables
            delete_duplicates_of_origin(self.goal_vae_encoder_variables)

        # for variable in self.network_variables:
        #     print(variable.name)
        # input("[enter]")


    def compute_gradients(self):
        print("\n---- COMPUTING GRADIENTS ----\n")

        if self.optimize_each:
            self.optimization = []

            def compute_vae_gradients(i, is_std_vae, encoder_variables, decoder_variables):
                var_list = []
                if is_std_vae:
                    var_list += decoder_variables
                    if not self.block_encoder_when_training_std_vae:
                        var_list += encoder_variables
                else:
                    var_list += self.network_variables
                    if not self.separate_std_vaes:
                        var_list += encoder_variables
                        if not self.sub_sequences_size-1 in self.std_vae_indices:
                            var_list += self.goal_vae_encoder_variables
                    if not self.block_decoder_when_training_transition:
                        var_list += decoder_variables

                cost = self.cost_per_vae[i]
                if self.optimize_like_rnn:
                    if i > 0:
                        cost = tf.add(tf.add_n(self.cost_per_vae[:i]), cost)
                    if not self.sub_sequences_size-1 in self.std_vae_indices and i != len(self.cost_per_vae)-1:
                        cost = tf.add(cost, self.cost_per_vae[-1])
                gradients = self.optimizer.compute_gradients(
                    cost,
                    # var_list=var_list
                )

                grads = np.array(gradients)[:, 0]
                variables = np.array(gradients)[:, 1]
                #for k, grad in enumerate(grads):
                   # if grad is not None:
                        #print(variables[k].name)
                # input("[enter]")

                self.optimization.append(self.optimizer.apply_gradients(gradients))

            # compute gradient for original vae
            compute_vae_gradients(
                0,
                True,
                self.origin_vae_encoder_variables,
                self.origin_vae_decoder_variables
            )
            print("\nOriginal VAE gradients computed.\n")

            offset = 1
            # compute gradients for all other std VAEs
            for i, _ in enumerate(self.std_vaes_like_origin):
                compute_vae_gradients(
                    offset+i,
                    True,
                    self.origin_vae_encoder_variables,
                    self.origin_vae_decoder_variables
                )
                print("\nVAE "+str(len(self.std_vae_indices) - len(self.std_vaes_like_origin) + i)+" sharing weights with origin gradients computed.\n")

            offset += len(self.std_vaes_like_origin)
            last_std_index = 0
            nb_model_vaes = 0
            for i in range(self.sub_sequences_size):
                if i not in self.std_vae_indices:
                    compute_vae_gradients(
                        offset+nb_model_vaes,
                        False,
                        self.origin_vae_encoder_variables,
                        self.origin_vae_decoder_variables
                    )
                    print("\nVAE subchain "+str(last_std_index)+" --> "+str(i)+" gradients computed.\n")
                    nb_model_vaes += 1
                else:
                    last_std_index = i

            offset += nb_model_vaes

            if not self.sub_sequences_size-1 in self.std_vae_indices:
                compute_vae_gradients(
                    offset,
                    True,
                    self.goal_vae_encoder_variables,
                    self.origin_vae_decoder_variables
                )
                print("\nGoal VAE gradients computed.\n")

            if self.z_continuity_error_coeff is not None and self.use_z_derivative:
                offset += 1
                var_list = []
                if self.std_vae_sequence_encoders:
                    var_list += self.origin_vae_encoder_variables
                else:
                    var_list += self.initial_z_derivative_variables

                gradients = self.optimizer.compute_gradients(
                    self.cost_per_vae[offset],
                    var_list=var_list
                )
                grads = np.array(gradients)[:, 1]
               # for k, variable in enumerate(grads):
                   # print(variable.name)
                # input("[enter]")

                self.optimization.append(self.optimizer.apply_gradients(gradients))
                print("\nInitial z derivative gradients computed.\n")
        else:
            def apply_gradients(gradients, proper_optimizer=False, optimizer_0=None):
                if optimizer_0 is not None:
                    optimizer = optimizer_0
                elif proper_optimizer:
                    optimizer = tf.train.AdamOptimizer(
                        learning_rate=self.learning_rate
                    )
                else:
                    optimizer = self.optimizer
                return optimizer.apply_gradients(gradients)

            def get_gradients(cost, proper_optimizer=False, optimizer_0=None, var_list=None):
                print("NEW GRADIENTS ----------")
                print("COST SHAPE = "+str(cost.get_shape()))
                if optimizer_0 is not None:
                    optimizer = optimizer_0
                elif proper_optimizer:
                    optimizer = tf.train.AdamOptimizer(
                        learning_rate=self.learning_rate
                    )
                else:
                    optimizer = self.optimizer
                gradients = optimizer.compute_gradients(cost, var_list=var_list)
                grads = np.array(gradients)[:, 0]
                variables = np.array(gradients)[:, 1]
                no_grad = True
                for k, grad in enumerate(grads):
                    if grad is not None:
                      #  print(variables[k].name)
                        no_grad = False
                if no_grad:
                    gradients = None
                return optimizer, gradients

            def extract_gradients(costs, size, proper_optimizer=False, var_list=None):
                opti = ()
                if proper_optimizer:
                    optimizer = tf.train.AdamOptimizer(
                        learning_rate=self.learning_rate
                    )
                else:
                    optimizer = None
                for i in range(size):
                    op = get_gradients(costs[i], optimizer_0=optimizer, var_list=var_list)
                    if op is not None:
                        opti += (op,)
                return opti

            def iteratate_over(costs_lists):
                opti = ()
                for costs in costs_lists:
                    opti += extract_gradients(costs)
                return opti

            nb_vae = len(self.cost_per_vae)
            self.optimization = ()
            op_and_gradients = []
            if self.separate_losses:
                variables = tf.trainable_variables()
                model_variables = [variable for variable in variables]
                for label in self.model.forbidden_gradient_labels:
                    model_variables = [variable for variable in model_variables if label not in variable.name]
                encoder_variables = [variable for variable in model_variables if "encoder" in variable.name]

                op_and_gradients.append(get_gradients(self.decoder_cost_per_dim_and_frame, var_list=variables))
                if not self.vae_only:
                    op_and_gradients.append(get_gradients(self.model_cost_per_dim_and_frame, var_list=model_variables, optimizer_0=op_and_gradients[0][0]))
                if self.z_continuity_error_coeff is not None:
                    op_and_gradients.append(get_gradients(self.continuity_cost_per_dim_and_frame, var_list=variables, optimizer_0=op_and_gradients[0][0]))
                op_and_gradients.append(get_gradients(self.encoder_cost_per_dim_and_frame, var_list=encoder_variables, optimizer_0=op_and_gradients[0][0]))
                # op_and_gradients.append(get_gradients(self.encoder_cost_per_dim_and_frame, var_list=variables, optimizer_0=op_and_gradients[0][0]))
            else:
                op_and_gradients.append(get_gradients(self.cost_per_dim_and_frame))

            for op, grad in op_and_gradients:
                if grad is not None:
                    self.optimization += (apply_gradients(grad, optimizer_0=op),)

        print("Done ------------------\n")


    def get_values(
        self,
        variables,
        Xs,
        transform_with_all_vtsfe=True,
        annealing_schedule=1.,
        fill=True,
        missing_value_indices=[],
        ignored_values=[[], []],
        nb_variables_per_frame=1.,
        input_dicts=[],
        recursive_call=False,
        nb_sub_sequences=None
    ): #"""Permet de récupérer les variables variables a partir de XS"""
        def fill_frames(subsequence, nb_params, sub_sequence_index):
            """ Fills the missing frames with nan (due to subsequence splitting)
            """
            # subsequences shape = [nb_params*sub_sequences_size, shape(val)]
            sub = np.array(subsequence)
            shape_val = sub.shape[1:]

            pre_shape = (int(nb_params*sub_sequence_index),) + shape_val
            pre = np.full(
                        pre_shape,
                        np.nan
                    )
            post_shape = (int(nb_params*(self.nb_frames - sub_sequence_index - self.sub_sequences_size)),) + shape_val
            post = np.full(
                        post_shape,
                        np.nan
                    )
            # subsequence_filled shape = [nb_params*nb_frames, shape(val)]
            subsequence_filled = np.concatenate([pre, subsequence, post], 0)
            return subsequence_filled
        
        transposed_Xs = np.transpose(Xs, axes=[1, 0, 2])
        values_filled_subs = []

        if transform_with_all_vtsfe:
            # If you want to retrieve your variables from the whole network

            if nb_sub_sequences is None:
                nss = self.nb_sub_sequences
            else:
                nss = nb_sub_sequences

            previous_dic = {}
            batch_size = len(Xs)
            zero = np.full((batch_size), 0, dtype=np.float32)
            #nz = dim of the latent space
            zero_z = np.full((batch_size, self.vae_architecture["n_z"]), 0, dtype=np.float32)
            zero_z_dim = np.full((self.vae_architecture["n_z"]), 0, dtype=np.float32)

            if self.z_continuity_error_coeff is not None:
                for i in range(self.sub_sequences_overlap):
                    previous_dic.update({
                        self.vae_subsequence[i].z_correction: zero_z_dim,
                        self.vae_subsequence[i].z_target: zero_z
                    })

                previous_dic.update({
                    self.origin.z_correction: zero_z_dim,
                    self.origin.z_target: zero_z
                })
                if not self.sub_sequences_size-1 in self.std_vae_indices:
                    previous_dic.update({
                        self.goal_vae.z_correction: zero_z_dim,
                        self.goal_vae.z_target: zero_z
                    })

                if self.use_z_derivative:
                    for i in range(self.sub_sequences_overlap):
                        previous_dic.update({
                            self.vae_subsequence[i].z_derivative_correction: zero_z_dim,
                            self.vae_subsequence[i].z_derivative_target: zero_z
                        })

                    previous_dic.update({
                        self.initial_z_derivative_correction: zero_z_dim,
                        self.initial_z_derivative_target: zero_z,
                        self.origin.z_derivative_correction: zero_z_dim,
                        self.origin.z_derivative_target: zero_z
                    })

            if self.model_params["model_continuity_error_coeff"] is not None:
                for i in range(self.sub_sequences_overlap):
                    if i not in self.std_vae_indices:
                        previous_dic.update({
                            self.vae_subsequence[i].var_correction: zero_z_dim,
                            self.vae_subsequence[i].var_target: zero_z
                        })

            for s in range(nss):
                sub_sequence_index = self.sub_sequences_step*s
                sub_sequence = Xs[:, sub_sequence_index : sub_sequence_index + self.sub_sequences_size]

                # reset f_dict
                f_dict = {
                    self.annealing_schedule: annealing_schedule,
                    # x_sequence shape = [nb_frames, batch_size, n_input]
                    self.x_sequence: transposed_Xs
                }

                f_dict[self.sub_sequence_origin] = sub_sequence_index

                for i in range(self.sub_sequences_size):
                    f_dict[self.x[i]] = sub_sequence[:, i]

                if input_dicts:
                    f_dict.update(input_dicts[s])

                f_dict.update(previous_dic)
                #variables :16 : tensor transpose (?,2) ; MUl (?,2) ; transpose (?,2) ; Mul (?,2) ;  [...]
                # values shape = [nb_values, shape(val)]

                #values.shape = [2.70.16]
                values = list(self.session.run(variables, feed_dict=f_dict))
                v = np.array(values)
                shape_val = v.shape[1:]

                # fills the missing values with nan
                for i in missing_value_indices:
                    for p in range(nb_variables_per_frame):
                        # values new shape = [nb_values + nb_missing_values*nb_variables_per_frame, shape(val)]
                        values.insert(nb_variables_per_frame*i, np.full(shape_val, np.nan))

                # replaces values from the ignored_values[1] frames by nan (only for subsequences in ignored_values[0])
                if s in ignored_values[0]:
                    for i in ignored_values[1]:
                        # values shape = [nb_variables_per_frame*sub_sequences_size, shape(val)]
                        values[int(nb_variables_per_frame*i)] = np.full(shape_val, np.nan)

                if fill:
                    # values shape = [nb_variables_per_frame*sub_sequences_size, shape(val)]
                    # values_filled shape = [nb_variables_per_frame*nb_frames, shape(val)]
                    values_filled = fill_frames(values, nb_variables_per_frame, sub_sequence_index)
                else:
                    values_filled = values

                # values_filled_subs shape = [nb_sub_sequences, -1, shape(val)]
                values_filled_subs.append(values_filled)

                # if we want to impose a continuity constraint on z or model between subsequences
                if self.z_continuity_error_coeff is not None or self.model_params["model_continuity_error_coeff"] is not None:
                    p = ()
                    index = self.sub_sequences_step

                    if self.z_continuity_error_coeff is not None:

                        for i in range(index, self.sub_sequences_size):
                            p += (self.vae_subsequence[i].z,)

                        z_derivative_indices = []
                        if self.use_z_derivative:
                            for i in range(index, self.sub_sequences_size):
                                if i not in self.std_vae_indices or self.std_vae_sequence_encoders or (self.initial_std_vae_only_sequence_encoder and i == 0):
                                    p += (self.vae_subsequence[i].z_derivative,)
                                    z_derivative_indices.append(i)

                        if self.absolute_last_frame:
                            p += (self.origin.z,)
                            if not self.sub_sequences_size-1 in self.std_vae_indices:
                                p += (self.goal_vae.z,)
                            if (self.std_vae_sequence_encoders or self.initial_std_vae_only_sequence_encoder) and self.use_z_derivative:
                                p += (self.origin.z_derivative,)

                    if self.model_params["model_continuity_error_coeff"] is not None:
                        for i in range(index, self.sub_sequences_size):
                            if i not in self.std_vae_indices:
                                p += (self.vae_subsequence[i].var_to_correct,)

                    values = list(self.session.run(p, feed_dict=f_dict))
                    previous_dic = {}
                    offset = 0
                    if self.z_continuity_error_coeff is not None:
                        for i in range(self.sub_sequences_overlap):
                            previous_dic.update({
                                self.vae_subsequence[i].z_target: values[i]
                            })

                        offset += self.sub_sequences_overlap

                        k = 0
                        if self.use_z_derivative:
                            for i in range(self.sub_sequences_overlap):
                                if i+index in z_derivative_indices:
                                    previous_dic.update({
                                        self.vae_subsequence[i].z_derivative_target: values[k]
                                    })
                                    k += 1
                                else:
                                    previous_dic.update({
                                        self.vae_subsequence[i].z_derivative_correction: zero_z_dim,
                                        self.vae_subsequence[i].z_derivative_target: zero_z
                                    })

                            if index in z_derivative_indices:
                                # in case initial_z_derivative_target and z_derivative_target from the first VAE of subsequece aren't the same
                                previous_dic[self.initial_z_derivative_target] = values[offset]
                            else:
                                # all in all, if you hadn't set z_derivative value for the first VAE of subsequence, you hadn't set initial_z_derivative_target
                                previous_dic.update({
                                    self.initial_z_derivative_target: zero_z
                                })

                        offset += len(z_derivative_indices)
                        if self.absolute_last_frame:
                            previous_dic.update({
                                self.origin.z_target: values[offset]
                            })
                            offset += 1
                            if not self.sub_sequences_size-1 in self.std_vae_indices:
                                previous_dic.update({
                                    self.goal_vae.z_target: values[offset]
                                })
                                offset += 1
                            if (self.std_vae_sequence_encoders or self.initial_std_vae_only_sequence_encoder) and self.use_z_derivative:
                                previous_dic.update({
                                    self.origin.z_derivative_target: values[offset],
                                })
                                offset += 1
                        else:
                            if not self.sub_sequences_size-1 in self.std_vae_indices:
                                previous_dic.update({
                                    self.goal_vae.z_correction: zero_z_dim,
                                    self.goal_vae.z_target: zero_z
                                })

                    if self.model_params["model_continuity_error_coeff"] is not None:
                        for i, k in enumerate(range(offset, offset + self.sub_sequences_overlap)):
                            if i not in self.std_vae_indices:
                                previous_dic.update({
                                    self.vae_subsequence[i].var_target: values[k]
                                })
        else:
            # If you want to retrieve your variables from only one standard VAE
            values = []

            for i in range(self.nb_frames):
                f_dict = {
                    self.annealing_schedule: annealing_schedule, # utilisé pour la prédiction : plus on connait d'epoch plus on est sur de la suite
                    # x_sequence shape = [nb_frames, batch_size, n_input]
                    self.x_sequence: transposed_Xs
                }

                if self.std_vae_sequence_encoders or self.initial_std_vae_only_sequence_encoder:
                    if i+self.input_sequence_size > self.nb_frames:
                        for k in range(self.input_sequence_size):
                            f_dict[self.x[k]] = Xs[:, i]
                    else:
                        for k in range(self.input_sequence_size):
                            f_dict[self.x[k]] = Xs[:, i+k]
                else:
                    f_dict[self.x[0]] = Xs[:, i]

                val = self.session.run(variables, feed_dict=f_dict)
                # values shape = [nb_frames, shape(val)]
                values.append(val)
            # values_filled_subs shape = [1, nb_frames, shape(val)]
            values_filled_subs.append(values)

        #retourne 70*70*2 = espace latent
        return values_filled_subs


    def get_values_from_latent_space_ori(
        self,
        variables,
        zs,
        transform_with_all_vtsfe=False,
        annealing_schedule=1.,
        fill=True,
        missing_value_indices=[],
        ignored_values=[[], []],
        nb_variables_per_frame=1.,
        recursive_call=False,
        nb_sub_sequences=None
    ):

        # If you want to retrieve your variables from only one standard VAE
        values = []

        #for i in range(self.nb_frames):
        f_dict = {
            self.annealing_schedule: annealing_schedule # utilisé pour la prédiction : plus on connait d'epoch plus on est sur de la suite
        }
        f_dict[self.vae_subsequence[0].z] = zs
        
        val = self.session.run(variables, feed_dict=f_dict)

        return val




    def partial_fit(self, Xs, annealing_schedule):
        """
        Train model based on mini-batch of input data.

        Return cost of mini-batch.
        """
        means = []
        variances = []
        std_vae_dicts = []
        means_goal = []
        variances_goal = []
        means_initial_z_derivative = []
        variances_initial_z_derivative = []

        def train_all_std_vaes(means, variances):
            # train all of the standard VAEs first and retrieve their original costs
            p = ()
            # compute gradient for original vae
            p += (self.optimization[0], self.cost_per_vae[0], self.var_per_vae[0], self.origin.z)

            offset = 1
            for i, vae in enumerate(self.std_vaes_like_origin):
                p += (self.optimization[offset+i], self.cost_per_vae[offset+i], self.var_per_vae[offset+i], vae.z)

            offset += len(self.std_vaes_like_origin)
            offset += self.sub_sequences_size - len(self.std_vae_indices)

            if not self.sub_sequences_size-1 in self.std_vae_indices:
                p += (self.optimization[offset], self.cost_per_vae[offset], self.var_per_vae[offset], self.goal_vae.z)

            if self.z_continuity_error_coeff is not None and self.use_z_derivative:
                p += (self.optimization[-1], self.cost_per_vae[-1], self.var_per_vae[-1])

            # values shape = [nb_sub_sequences, nb_std_vaes*3*shape(val)]
            values = self.get_values(p, Xs, annealing_schedule=annealing_schedule, fill=False)
            values = np.array(values)

            costs = np.array(values[:, 1::4], dtype=np.float32)
            vs = values[:, 2::4]
            zs = np.array([[np.array(y, dtype=np.float32) for y in x] for x in values[:, 3::4]], dtype=np.float32)

            offset = 1 + len(self.std_vaes_like_origin)
            if not self.sub_sequences_size-1 in self.std_vae_indices:
                means_goal = costs[:, offset]
                variances_goal = vs[:, offset]
            else:
                means_goal = None
                variances_goal = None
            if self.z_continuity_error_coeff is not None and self.use_z_derivative:
                means_initial_z_derivative = costs[:, -1]
                variances_initial_z_derivative = vs[:, -1]
            else:
                means_initial_z_derivative = []
                variances_initial_z_derivative = []

            for i in range(len(self.std_vaes_like_origin)+1):
                # costs shape = [sub_sequences_size+(2 or 1), nb_sub_sequences]
                means.append(costs[:, i])
                variances.append(vs[:, i])
            if True in np.isinf(zs) or True in np.isinf(costs):
                print("INF !")
                # nb_z = len(zs[0])
                # print("costs :")
                # print(costs)
                # _ = input("Press [enter] to continue.") # wait for input from the user
                # for k in range(nb_z):
                #     print("z value :")
                #     print(zs[:, k])
                #     _ = input("Press [enter] to continue.") # wait for input from the user
                self.crashed = True
            if True in np.isnan(zs) or True in np.isnan(costs):
                print("NAN !")
                # nb_z = len(zs[0])
                # print("costs :")
                # print(costs)
                # _ = input("Press [enter] to continue.") # wait for input from the user
                # for k in range(nb_z):
                #     print("z value :")
                #     print(zs[:, k])
                #     _ = input("Press [enter] to continue.") # wait for input from the user
                self.crashed = True
            return (means_goal, variances_goal, means_initial_z_derivative, variances_initial_z_derivative)

        if self.optimize_each:
            input_dicts = []
            input_dicts += std_vae_dicts

            # train all of the standard VAEs first and retrieve their original costs
            means_goal, variances_goal, means_initial_z_derivative, variances_initial_z_derivative = train_all_std_vaes(means, variances)

            offset = 1
            offset += len(self.std_vaes_like_origin)
            nb_model_vaes = 0
            for i, vae in enumerate(self.vae_subsequence):
                if i in self.std_vae_indices:
                    continue

                # train that particular VAE first and retrieve its original cost
                p = (self.optimization[offset+nb_model_vaes], self.cost_per_vae[offset+nb_model_vaes], self.var_per_vae[offset+nb_model_vaes],
                    vae.z, vae.sys_noise, vae.sys_noise_log_sigma_sq, vae.f)

                # values shape = [nb_sub_sequences, 2, shape(val)]
                values = self.get_values(p, Xs, annealing_schedule=annealing_schedule, fill=False, input_dicts=input_dicts)
                # values = np.array(values, dtype=np.float32)
                values = np.array(values)
                # means shape = [sub_sequences_size, nb_sub_sequences]
                means.append(values[:, 1])
                variances.append(values[:, 2])
                vrbls = np.array([[np.array(y, dtype=np.float32) for y in x] for x in values[:, 3:]], dtype=np.float32)

                costs = np.array(values[:, 1], dtype=np.float32)
                if True in np.isinf(vrbls) or True in np.isinf(costs):
                    print("INF !")
                    # print("costs :")
                    # print(costs)
                    # _ = input("Press [enter] to continue.") # wait for input from the user
                    # for k in range(1, 7):
                    #     print(str(k)+" value :")
                    #     print(values[:, k])
                    #     _ = input("Press [enter] to continue.") # wait for input from the user
                    self.crashed = True
                if True in np.isnan(vrbls) or True in np.isnan(costs):
                    print("NAN !")
                    # print("costs :")
                    # print(costs)
                    # _ = input("Press [enter] to continue.") # wait for input from the user
                    # for k in range(1, 7):
                    #     print(str(k)+" value :")
                    #     print(values[:, k])
                    #     _ = input("Press [enter] to continue.") # wait for input from the user
                    self.crashed = True

                if self.bypass_back_prop_through_time:
                    # then get optimized z and z_derivative values
                    p = (vae.z, vae.z_derivative,)

                    # values shape = [nb_sub_sequences, 2, shape(val)]
                    values = self.get_values(p, Xs, annealing_schedule=annealing_schedule, fill=False, input_dicts=input_dicts)
                    values = np.array(values, dtype=np.float32)

                    # and add obtained values to bypass back-propagation of the next VAE
                    input_dicts = []
                    for s in range(self.nb_sub_sequences):
                        dic = {}
                        if std_vae_dicts:
                            dic.update(std_vae_dicts[s])
                        dic.update({
                            vae.z: values[s, 0],
                            vae.z_derivative: values[s, 1]
                        })
                        input_dicts.append(dic)

                nb_model_vaes += 1

            # means shape = [nb_sub_sequences, sub_sequences_size]
            means = np.transpose(means)
            variances = np.transpose(variances)
            # global_cost shape = [nb_sub_sequences]
            global_cost = np.mean(means, 1)
            # global_cost = np.divide(np.sum(means, 0), len(self.cost_per_vae))
        else:
            if self.separate_std_vaes:
                # train all of the standard VAEs first and retrieve their original costs
                means_goal, variances_goal, means_initial_z_derivative, variances_initial_z_derivative = train_all_std_vaes(means, variances)

            p = (self.cost,) + self.optimization
            offset = len(p)
            jump = 0
            for i in range(len(self.cost_per_vae)):
                costs = (
                    self.cost_per_vae[i], self.var_per_vae[i],
                    self.reconstr_cost_per_vae[i], self.reconstr_var_per_vae[i],
                    self.model_cost_per_vae[i], self.model_var_per_vae[i],
                    self.latent_cost_per_vae[i], self.latent_var_per_vae[i],
                )
                if i == 0:
                    jump = len(costs)
                p += costs

            values = self.get_values(p, Xs, annealing_schedule=annealing_schedule, fill=False, input_dicts=std_vae_dicts)
            values = np.array(values, dtype=np.float32)

            if True in np.isinf(values[:, 0]):
                print("INF cost !")
                self.crashed = True
            if True in np.isnan(values[:, 0]):
                print("NAN cost !")
                self.crashed = True

            # means shape = [nb_sub_sequences, sub_sequences_size]
            means = values[:, offset::jump]
            variances = values[:, offset+1::jump]

            reconstr_means = values[:, offset+2::jump]
            reconstr_variances = values[:, offset+3::jump]

            model_means = values[:, offset+4::jump]
            model_variances = values[:, offset+5::jump]

            latent_means = values[:, offset+6::jump]
            latent_variances = values[:, offset+7::jump]
            # global_cost shape = [nb_sub_sequences]
            global_cost = values[:, 0]

        if np.array(means_goal).any():
            # at the end, add goal data to means and variances to always keep the same VAE order
            means = np.concatenate([means, np.reshape(means_goal, [-1, 1])], axis=1)
            variances = np.concatenate([variances, np.reshape(variances_goal, [-1, 1])], axis=1)

            if self.z_continuity_error_coeff is not None and self.use_z_derivative:
                means = np.concatenate([means, np.reshape(means_initial_z_derivative, [-1, 1])], axis=1)
                variances = np.concatenate([variances, np.reshape(variances_initial_z_derivative, [-1, 1])], axis=1)

        # global_cost shape = [nb_sub_sequences], means and variances shape = [nb_sub_sequences, nb_vaes]
        return (
            global_cost,
            means, variances,
            reconstr_means, reconstr_variances,
            model_means, model_variances,
            latent_means, latent_variances
        )


    def transform(self, Xs, transform_with_all_vtsfe=True):
        """ Transform data by mapping it into the latent space."""

        p = ()

        if transform_with_all_vtsfe:
            # If you want to retrieve your variables from the whole network
            for i, vae in enumerate(self.vae_subsequence):
                p += (vae.z,)
        else:
            # If you want to retrieve your variables from only one standard VAE
            p = self.vae_subsequence[0].z
        #print(Xs.shape) #(70, 70, 66)
        #print(p)  #Tensor("Add_92:0", shape=(?, 2), dtype=float32)
        
        values = self.get_values(p, Xs, transform_with_all_vtsfe=transform_with_all_vtsfe)
        #values = 1*70*70*2
        return values


    def transform_derivative(self, Xs):
        """ Transform data by mapping it into the latent space."""

        p = ()
        no_z_derivative_indices = []

        for i, vae in enumerate(self.vae_subsequence):
            if i not in self.std_vae_indices or self.std_vae_sequence_encoders or (self.initial_std_vae_only_sequence_encoder and i == 0):
                p += (vae.z_derivative,)
            else:
                no_z_derivative_indices.append(i)

        # values shape = [nb_frames-len(no_z_derivative_indices), batch_size, n_z]
        values = self.get_values(p, Xs, transform_with_all_vtsfe=True)
        shape_val = values[0].shape
        for i in no_z_derivative_indices:
            np.insert(values, i, np.full(shape_val, np.nan))

        return values


    def reconstruct(self, Xs, transform_with_all_vtsfe=True, fill=True):
        """Reconstruct data"""

        p = ()
        if transform_with_all_vtsfe:
            # If you want to retrieve your variables from the whole network
            for i, vae in enumerate(self.vae_subsequence):
                p += (vae.x_reconstr,)
        else:
            # If you want to retrieve your variables from only one standard VAE
            p = self.vae_subsequence[0].x_reconstr
        #Xs : (70, 70, 66)  
 
        #p[0]Tensor("strided_slice_211:0", shape=(66,), dtype=float32)
        # p[1]Tensor("strided_slice_212:0", shape=(66,), dtype=float32)
        print(p) #Tensor("Identity_3:0", shape=(?, 66), dtype=float32)
        #values = 1.70.70.66

        values = self.get_values(p, Xs, transform_with_all_vtsfe=transform_with_all_vtsfe, fill=fill)


        return values

    def reconstruct_fromLS(self, zs, transform_with_all_vtsfe=False, fill=True):
        """Reconstruct data from zs"""
  
        p = ()

        if transform_with_all_vtsfe:
            # If you want to retrieve your variables from the whole network
            for i, vae in enumerate(self.vae_subsequence):
                p += (vae.x_reconstr,)
        else:
            # If you want to retrieve your variables from only one standard VAE
            p = self.vae_subsequence[0].x_reconstr

        values = self.get_values_from_latent_space_ori(p, zs, transform_with_all_vtsfe=False, fill=fill)

   
        return values


    def from_subsequences_to_sequence(self, x_samples, x_reconstr):
        # transform the subsequences of x_reconstr in one sequence
        step = 0
        step += self.sub_sequences_step
        x_seq = [x for x in x_reconstr[0, :self.sub_sequences_step]]
        x_seq_weights = [1 for x in x_reconstr[0, :self.sub_sequences_step]]
        # for i, _ in enumerate(x_seq):
            # print(i)
        for s in range(len(x_reconstr)-1):
            # print("s_seq length = "+str(len(x_seq)))
            # print("overlapping part")
            for i in range(self.sub_sequences_overlap):
                index_abs = step+i
                # print(index_abs)
                index_rel = self.sub_sequences_step+i
                if index_abs < len(x_seq):
                    x_seq_weights[index_abs] += 1
                    w = 1./x_seq_weights[index_abs]
                    x_seq[index_abs] = (1.-w)*x_seq[index_abs] + w*x_reconstr[s+1, i]
                else:
                    mean = (x_reconstr[s, index_rel] + x_reconstr[s+1, i])/2
                    x_seq.append(mean)
                    x_seq_weights.append(2)
            # print("simple part")
            for k, x in enumerate(x_reconstr[s+1, self.sub_sequences_overlap:self.sub_sequences_step]):
                # print(self.sub_sequences_overlap+step+k)
                x_seq.append(x)
            step += self.sub_sequences_step
            # print("s_seq length = "+str(len(x_seq)))
        # print("end")
        remains_index = max(self.sub_sequences_step, self.sub_sequences_overlap)
        step = step - self.sub_sequences_step + remains_index
        for k, x in enumerate(x_reconstr[-1, remains_index:]):
            # print(step+k)
            x_seq.append(x)
        # print(len(x_seq))
        # in case the number of frames isn't divisible by the computed number of subsequences, add the original rest to the reconstructed sequence
        for x in np.transpose(x_samples, [1, 0, 2])[len(x_seq):]:
            x_seq.append(x)
        # x_seq shape [nb_frames, nb_samples, n_input]
        return x_seq


    def get_reconstruction_squarred_error(self, data, reconstruction, transform_with_all_vtsfe=True):
        # reconstruction shape = [nb_sub_sequences, sub_sequences_size, nb_samples, n_input]
        if transform_with_all_vtsfe:
            x_seq = self.from_subsequences_to_sequence(data, reconstruction)
            # transpose x_seq to shape [nb_samples, nb_frames, n_input]
            x_seq = np.transpose(x_seq, [1, 0, 2])
        else:
            x_seq = np.reshape(reconstruction, [-1, self.nb_frames, self.vae_architecture["n_input"]])

        return np.square(np.subtract(data, x_seq))


    def init_session(self):
        if not self.restore or self.save_path is None or not os.path.isfile("./checkpoints/"+self.save_path+'.index'):
            # Initializing the tensor flow variables
            self.session.run(tf.global_variables_initializer())
        else:
            # Restore variables from disk
            tf.train.Saver().restore(self.session, "./checkpoints/"+self.save_path)
            print("\n---- VTSFE "+self.save_path+" RESTORED ----\n")


    def save_session(self, data_driver, data, epoch ):
        # Save the variables to disk.
        #pdb.set_trace()
        if self.save_path is not None:
            tf.train.Saver().save(self.session, "./checkpoints/"+self.save_path+"_epoch_"+str(epoch))
            data_driver.save_data("./training_errors/"+self.save_path+"_epoch_"+str(epoch), "errors", data)

            print("\n---- VTSFE SAVED IN FILE: ./checkpoints/"+self.save_path+"_epoch_"+str(epoch) + " ----\n")
            print("\n---- TRAINING ERRORS SAVED IN FILE: ./training_errors/"+self.save_path+"_epoch_"+str(epoch)+" ----\n")


    def train(self, data, training):
        # data shape = [nb_samples, nb_frames, dim(values)]
        n_samples = len(data)
        annealing_schedule_temp = training["annealing_schedule_temp"]
        self.crashed = False

        with self.session.as_default():
            # Training cycle

            begin = datetime.now()
            print("\n------- Training begining: {} -------\n".format(begin.isoformat()[11:]))
            print("Number of samples = "+str(n_samples))
            print("Batch size = "+str(training["batch_size"]))
            print("Sequence size = "+str(self.nb_frames))
            print("Subsequence size = "+str(self.sub_sequences_size))
            print("Number of subsequences = "+str(self.nb_sub_sequences)+"\n")
            print("Latent space dimension = "+str(self.vae_architecture["n_z"])+"\n")

            global_error = []
            vae_errors = []
            vae_variances = []
            vae_reconstr_errors = []
            vae_reconstr_variances = []
            vae_model_errors = []
            vae_model_variances = []
            vae_latent_errors = []
            vae_latent_variances = []

            if self.restore:
                (global_error,
                vae_errors, vae_variances,
                vae_reconstr_errors, vae_reconstr_variances,
                vae_model_errors, vae_model_variances,
                vae_latent_errors, vae_latent_variances) = training["data_driver"].read_data("./training_errors/"+self.save_path, "errors")
            for epoch in range(training["nb_epochs"]):
                nb_batches = int(n_samples / training["batch_size"])
                remains_batch_size = n_samples % training["batch_size"]
                if remains_batch_size != 0:
                    nb_batches += 1

                if annealing_schedule_temp is not None and annealing_schedule_temp != 0:
                    annealing_schedule = min(1., 1E-2 + epoch / annealing_schedule_temp)
                    # annealing_schedule = max(1., annealing_schedule_temp / (1E-2 + epoch))
                    # annealing_schedule = max(1., 1E-2 + epoch / annealing_schedule_temp)
                else:
                    annealing_schedule = 1.

                w = 1.
                nb_samples = 0
                # 1 origin
                nb_vaes = 1
                if not self.sub_sequences_size-1 in self.std_vae_indices:
                    # 1 goal
                    nb_vaes += 1
                nb_vaes += len(self.std_vaes_like_origin) + self.sub_sequences_size - len(self.std_vae_indices)

                avg_cost = 0.
                avg_reconstr_cost = 0.
                avg_model_cost = 0.
                avg_latent_cost = 0.
                avg_errors = np.full(nb_vaes, 0.)
                avg_reconstr_errors = np.full(nb_vaes, 0.)
                avg_model_errors = np.full(nb_vaes, 0.)
                avg_latent_errors = np.full(nb_vaes, 0.)
                intra_batch_variances = np.full(nb_vaes, 0.)
                intra_batch_reconstr_variances = np.full(nb_vaes, 0.)
                intra_batch_model_variances = np.full(nb_vaes, 0.)
                intra_batch_latent_variances = np.full(nb_vaes, 0.)
                inter_batch_variances = np.full(nb_vaes, 0.)
                inter_batch_reconstr_variances = np.full(nb_vaes, 0.)
                inter_batch_model_variances = np.full(nb_vaes, 0.)
                inter_batch_latent_variances = np.full(nb_vaes, 0.)
                batch_errors = []
                batch_reconstr_errors = []
                batch_model_errors = []
                batch_latent_errors = []

                # Loop over all batches
                for i in range(nb_batches):
                    cost = 0.
                    batch = data[training["batch_size"]*i : training["batch_size"]*(i+1)]
                    # Fit training using batch data and average cost on all subsequences
                    # batch_cost shape = [nb_sub_sequences], means and variances shape = [nb_sub_sequences, nb_vaes]
                    (
                        batch_cost,
                        batch_error_means, batch_error_variances,
                        batch_error_reconstr_means, batch_error_reconstr_variances,
                        batch_error_model_means, batch_error_model_variances,
                        batch_error_latent_means, batch_error_latent_variances
                    ) = self.partial_fit(batch, annealing_schedule)

                    if self.crashed:
                        break

                    # averages over subsequences
                    bem_tot = np.mean(batch_error_means, 0)
                    bev_tot = np.add(
                        np.mean(batch_error_variances, 0),
                        np.var(batch_error_means, 0)
                    )
                    berm_tot = np.mean(batch_error_reconstr_means, 0)
                    berv_tot = np.add(
                        np.mean(batch_error_reconstr_variances, 0),
                        np.var(batch_error_means, 0)
                    )
                    bemm_tot = np.mean(batch_error_model_means, 0)
                    bemv_tot = np.add(
                        np.mean(batch_error_model_variances, 0),
                        np.var(batch_error_means, 0)
                    )
                    belm_tot = np.mean(batch_error_latent_means, 0)
                    belv_tot = np.add(
                        np.mean(batch_error_latent_variances, 0),
                        np.var(batch_error_means, 0)
                    )

                    bc_mean = np.mean(batch_cost, 0)
                    brc_mean = np.mean(berm_tot, 0)
                    bmc_mean = np.mean(bemm_tot, 0)
                    blc_mean = np.mean(belm_tot, 0)

                    # average over batches
                    batch_errors.append(bc_mean)
                    batch_reconstr_errors.append(brc_mean)
                    batch_model_errors.append(bmc_mean)
                    batch_latent_errors.append(blc_mean)

                    # Compute average loss
                    nb_samples += len(batch)
                    w = len(batch)/nb_samples
                    avg_cost = avg_cost*(1. - w) + w*bc_mean
                    avg_reconstr_cost = avg_reconstr_cost*(1. - w) + w*brc_mean
                    avg_model_cost = avg_model_cost*(1. - w) + w*bmc_mean
                    avg_latent_cost = avg_latent_cost*(1. - w) + w*blc_mean

                    # Compute average loss and variance on each VAE
                    for k in range(nb_vaes):
                        avg_errors[k] = avg_errors[k]*(1. - w) + w*bem_tot[k]
                        intra_batch_variances[k] = intra_batch_variances[k]*(1. - w) + w*bev_tot[k]
                        avg_reconstr_errors[k] = avg_reconstr_errors[k]*(1. - w) + w*berm_tot[k]
                        intra_batch_reconstr_variances[k] = intra_batch_reconstr_variances[k]*(1. - w) + w*berv_tot[k]
                        avg_model_errors[k] = avg_model_errors[k]*(1. - w) + w*bemm_tot[k]
                        intra_batch_model_variances[k] = intra_batch_model_variances[k]*(1. - w) + w*bemv_tot[k]
                        avg_latent_errors[k] = avg_latent_errors[k]*(1. - w) + w*belm_tot[k]
                        intra_batch_latent_variances[k] = intra_batch_latent_variances[k]*(1. - w) + w*belv_tot[k]

                if self.crashed:
                    break

                global_error.append(avg_cost)
                vae_errors.append(avg_errors)
                vae_reconstr_errors.append(avg_reconstr_errors)
                vae_model_errors.append(avg_model_errors)
                vae_latent_errors.append(avg_latent_errors)

                # Compute also inter-batch variance
                nb_samples = remains_batch_size
                if nb_samples == 0:
                    nb_samples = training["batch_size"]
                w = 1.
                # go through batches from the last one to the first one
                for b in range(nb_batches-1, -1, -1):
                    for k in range(nb_vaes):
                        inter_batch_variances[k] = inter_batch_variances[k]*(1. - w) + w*np.square(batch_errors[b] - avg_cost)
                        inter_batch_reconstr_variances[k] = inter_batch_reconstr_variances[k]*(1. - w) + w*np.square(batch_reconstr_errors[b] - avg_reconstr_cost)
                        inter_batch_model_variances[k] = inter_batch_model_variances[k]*(1. - w) + w*np.square(batch_model_errors[b] - avg_model_cost)
                        inter_batch_latent_variances[k] = inter_batch_latent_variances[k]*(1. - w) + w*np.square(batch_latent_errors[b] - avg_latent_cost)
                    nb_samples += training["batch_size"]
                    w = training["batch_size"]/nb_samples

                # Compute the total variance per VAE
                total_variances = np.add(intra_batch_variances, inter_batch_variances)
                total_reconstr_variances = np.add(intra_batch_reconstr_variances, inter_batch_reconstr_variances)
                total_model_variances = np.add(intra_batch_model_variances, inter_batch_model_variances)
                total_latent_variances = np.add(intra_batch_latent_variances, inter_batch_latent_variances)
                vae_variances.append(total_variances)
                vae_reconstr_variances.append(total_reconstr_variances)
                vae_model_variances.append(total_model_variances)
                vae_latent_variances.append(total_latent_variances)

                # Display logs per epoch step
                if epoch % training["display_step"] == 0 or epoch == training["nb_epochs"]-1:
                    print("Epoch:", '%04d' % (epoch),
                        " -------> Average cost =", "{:.9f}".format(avg_cost),
                        " | Average reconstruction cost =", "{:.9f}".format(avg_reconstr_cost),
                        " | Average model cost =", "{:.9f}".format(avg_model_cost),
                        " | Average latent cost =", "{:.9f}".format(avg_latent_cost))

                if epoch > 0 and epoch % training["checkpoint_step"] == 0:
                    self.save_session(training["data_driver"], (
                        global_error,
                        vae_errors, vae_variances,
                        vae_reconstr_errors, vae_reconstr_variances,
                        vae_model_errors, vae_model_variances,
                        vae_latent_errors, vae_latent_variances
                    ),epoch)

            end = datetime.now()
            print("\n------- Training end: {} -------\n".format(end.isoformat()[11:]))
            print("Elapsed = "+str((end-begin).total_seconds())+" seconds\n")

            self.save_session(training["data_driver"], (
                global_error,
                vae_errors, vae_variances,
                vae_reconstr_errors, vae_reconstr_variances,
                vae_model_errors, vae_model_variances,
                vae_latent_errors, vae_latent_variances
            ),epoch)
            return (
                global_error,
                vae_errors, vae_variances,
                vae_reconstr_errors, vae_reconstr_variances,
                vae_model_errors, vae_model_variances,
                vae_latent_errors, vae_latent_variances,
                epoch
            )



    def compute_stats(self, data_driver, reconstr_datasets, reconstr_datasets_names, sample_indices,x_samples, nb_samples_per_mov=1, transform_with_all_vtsfe=False, data_inf=[]):
        # don't display more movements than there are
        if nb_samples_per_mov > len(sample_indices):
            display_per_mov = len(sample_indices)
        else:
            display_per_mov = nb_samples_per_mov
        print(str(display_per_mov))
        if transform_with_all_vtsfe:
            nb_sub_sequences = self.nb_sub_sequences
        else:
            nb_sub_sequences = 1

        data = np.copy(x_samples) #70.70.66

        data_reconstr = []
        for j,reco in enumerate(reconstr_datasets): #1.1.70.70.66
            data_reconstr.append(reco.reshape([nb_sub_sequences, len(data_driver.mov_types)*data_driver.nb_samples_per_mov, self.nb_frames, -1, 3]))
        segment_count = len(data[0, 0])#22

        data_reconstr2 = reconstr_datasets[0][0]

        my_statistics = My_statistics(data, data_inf, data_reconstr2, [10,70,70,69]) #TODO rendre 5/10 en variable
        my_statistics.get_distance();

        return  my_statistics
         

    def show_data(self, params):
        self.vtsfe_plots.show_data(**params)
        #self.vtsfe_plots.show_data_ori2(**params)

    def show_data_ori(self,x_mean, params):
        #self.vtsfe_plots.show_data(**params)
        self.vtsfe_plots.show_data_ori(x_mean, **params)

    def plot_mse(self, params):

        self.vtsfe_plots.plot_mse(**params)


    def plot_error(self, errors):
        self.vtsfe_plots.plot_error(*errors)


    def show_latent_space(self, data_driver, latent_data,  sample_indices, title, zs_inf=[], displayed_movs=[], nb_samples_per_mov=1, show_frames=False, titleFrame=None):
      #  if (len(latent_data)==1):
        self.vtsfe_plots.show_latent_space(data_driver, latent_data, sample_indices, title,zs_inf, displayed_movs=displayed_movs, nb_samples_per_mov=nb_samples_per_mov, show_frames=show_frames, titleFrame=titleFrame)
        #else:
            #self.vtsfe_plots.show_latent_space_ori(data_driver, latent_data, zs_inf, sample_indices, title, show_frames=show_frames)

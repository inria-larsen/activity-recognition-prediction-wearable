# -*- coding: utf-8 -*-

from datetime import datetime
import os.path
import operator

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

from .lib.useful_functions import *


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

        if len(self.std_vae_indices) < self.sub_sequences_size:
            # if there is model VAEs in the VTSFE network
            self.model = self.model_class(self.model_params, self)

        if self.input_sequence_size > 1 and not self.initial_std_vae_only_sequence_encoder:
            self.std_vae_sequence_encoders = True
        else:
            self.std_vae_sequence_encoders = False

        self.devices = get_available_gpus()

        # Create a chain of vaes
        self.create_network()
        # Extra-wiring specific to model
        self.model.model_wiring()

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
        # self.writer = tf.summary.FileWriter("./log_dir", self.session.graph)

        # init session to restore or randomly initialize all variables in network
        self.init_session()

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
            # if we want an absolute last frame reference, then you have to create an additional std VAE for an absolute first frame reference
            edge_vae_data_tensor = self.x_sequence
        else:
            # otherwise, just create the first std VAE of subsequence
            edge_vae_data_tensor = self.x

        if self.std_vae_sequence_encoders or self.initial_std_vae_only_sequence_encoder:
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
                    if self.z_continuity_error_coeff != None and i < self.sub_sequences_overlap:
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
            else:
                self.vae_subsequence.append(None)

        if self.use_z_derivative:
            self.estimate_initial_z_derivative()

        print("\nVTSFE NETWORK ---- INITIATE "+str(self.sub_sequences_size - len(self.std_vae_indices))+" MODEL VAEs :")

        transition_initiated = False
        for i in range(self.sub_sequences_size):
            if i not in self.std_vae_indices:
                print("---------- "+str(i))

                if transition_initiated:
                    reuse_encoder_weights = True
                else:
                    reuse_encoder_weights = False
                    transition_initiated = True

                z_continuity_error_coeff = None
                model_continuity_error_coeff = None
                if i < self.sub_sequences_overlap:
                    if self.z_continuity_error_coeff != None:
                        z_continuity_error_coeff = self.z_continuity_error_coeff

                    if self.model_params["model_continuity_error_coeff"] != None:
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

            if self.z_continuity_error_coeff != None:
                self.initial_z_derivative_target = self.vae_subsequence[0].z_derivative_target
                self.initial_z_derivative_correction = self.vae_subsequence[0].z_derivative_correction
                self.reduced_initial_z_derivative_correction = self.vae_subsequence[0].reduced_z_derivative_correction
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

            if self.z_continuity_error_coeff != None:
                self.initial_z_derivative_target = tf.placeholder(tf.float32, shape=[None, self.vae_architecture["n_z"]], name="initial_z_derivative_target")
                self.initial_z_derivative_correction = tf.square(
                    tf.subtract(
                        z_derivative,
                        self.initial_z_derivative_target
                    )
                )
                self.reduced_initial_z_derivative_correction = tf.scalar_mul(
                    tf.constant(self.z_continuity_error_coeff, dtype=tf.float32),
                    tf.reduce_sum(
                        self.initial_z_derivative_correction,
                        1
                    )
                )

            print("Z DERIVATIVE ESTIMATION NETWORK ---- END\n")

        self.initial_z_derivative = z_derivative
        if self.z_continuity_error_coeff != None:
            self.initial_z_derivative_cost, self.initial_z_derivative_variance = tf.nn.moments(self.reduced_initial_z_derivative_correction, [0], name="initial_z_derivative_cost_and_variance")


    def get_costs_and_variances(self):
        self.cost_per_vae = []
        self.var_per_vae = []

        def get_cost_and_variance(vae):
            m, v = vae.cost, vae.variance
            # self.cost_per_vae shape = [sub_sequences_size]
            self.cost_per_vae.append(m)
            # self.var_per_vae shape = [sub_sequences_size]
            self.var_per_vae.append(v)

        get_cost_and_variance(
            self.origin
        )

        for vae in self.std_vaes_like_origin:
            get_cost_and_variance(
                vae
            )

        for i, vae in enumerate(self.vae_subsequence):
            if i not in self.std_vae_indices:
                get_cost_and_variance(vae)

        if not self.sub_sequences_size-1 in self.std_vae_indices:
            get_cost_and_variance(
                self.goal_vae
            )

        if self.z_continuity_error_coeff != None and self.use_z_derivative:
            m, v = self.initial_z_derivative_cost, self.initial_z_derivative_variance
            self.cost_per_vae.append(m)
            self.var_per_vae.append(v)


    def create_loss_function(self):
        """
        Creates a global loss function for VTSFE based on each VAE loss function
        """
        costs = []

        def get_cost(vae, costs):
            costs.append(vae.cost_add)

        if not self.separate_std_vaes:
            get_cost(
                self.origin,
                costs
            )

            for vae in self.std_vaes_like_origin:
                get_cost(
                    vae,
                    costs
                )

            if not self.sub_sequences_size-1 in self.std_vae_indices:
                get_cost(
                    self.goal_vae,
                    costs
                )

            if self.z_continuity_error_coeff != None and self.use_z_derivative:
                costs.append(self.reduced_initial_z_derivative_correction)

        for i, vae in enumerate(self.vae_subsequence):
            if i not in self.std_vae_indices:
                get_cost(vae, costs)

        # sum on all frames
        cost = tf.add_n(costs)
        # average over batch
        self.cost = tf.reduce_mean(cost, 0)
        # self.costs = tf.reduce_mean(costs, 1)


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
        self.origin_vae_encoder_variables = get_sorted_values_from_dict(origin_vae_variables["weights_encoder"], [])
        self.origin_vae_encoder_variables = get_sorted_values_from_dict(origin_vae_variables["biases_encoder"], self.origin_vae_encoder_variables)

        # get origin VAE decoder variables
        self.origin_vae_decoder_variables = get_sorted_values_from_dict(origin_vae_variables["weights_decoder"], [])
        self.origin_vae_decoder_variables = get_sorted_values_from_dict(origin_vae_variables["biases_decoder"], self.origin_vae_decoder_variables)

        if not self.sub_sequences_size-1 in self.std_vae_indices:
            # get goal VAE encoder variables (might be different from the origin VAE ones in case of usage of sequence encoders)
            self.goal_vae_encoder_variables = get_sorted_values_from_dict(goal_vae_variables["weights_encoder"], [])
            self.goal_vae_encoder_variables = get_sorted_values_from_dict(goal_vae_variables["biases_encoder"], self.goal_vae_encoder_variables)

        # get all remaining variables in network
        for dic in model_variables.values():
            self.network_variables = get_sorted_values_from_dict(dic, self.network_variables)

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
                for k, grad in enumerate(grads):
                    if grad != None:
                        print(variables[k].name)
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

            if self.z_continuity_error_coeff != None and self.use_z_derivative:
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
                for k, variable in enumerate(grads):
                    print(variable.name)
                # input("[enter]")

                self.optimization.append(self.optimizer.apply_gradients(gradients))
                print("\nInitial z derivative gradients computed.\n")
        else:
            gradients = self.optimizer.compute_gradients(self.cost)
            # gradients = self.optimizer.compute_gradients(self.cost)

            grads = np.array(gradients)[:, 0]
            variables = np.array(gradients)[:, 1]
            for k, grad in enumerate(grads):
                if grad != None:
                    print(variables[k].name)
            self.optimization = self.optimizer.apply_gradients(gradients)

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
    ):
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

            if nb_sub_sequences == None:
                nss = self.nb_sub_sequences
            else:
                nss = nb_sub_sequences

            previous_dic = {}
            batch_size = len(Xs)
            zero = np.full((batch_size), 0, dtype=np.float32)
            zero_z = np.full((batch_size, self.vae_architecture["n_z"]), 0, dtype=np.float32)

            if self.z_continuity_error_coeff != None:
                for i in range(self.sub_sequences_overlap):
                    previous_dic.update({
                        self.vae_subsequence[i].reduced_z_correction: zero,
                        self.vae_subsequence[i].z_target: zero_z
                    })

                previous_dic.update({
                    self.origin.reduced_z_correction: zero,
                    self.origin.z_target: zero_z
                })
                if not self.sub_sequences_size-1 in self.std_vae_indices:
                    previous_dic.update({
                        self.goal_vae.reduced_z_correction: zero,
                        self.goal_vae.z_target: zero_z
                    })

                if self.use_z_derivative:
                    for i in range(self.sub_sequences_overlap):
                        previous_dic.update({
                            self.vae_subsequence[i].reduced_z_derivative_correction: zero,
                            self.vae_subsequence[i].z_derivative_target: zero_z
                        })

                    previous_dic.update({
                        self.reduced_initial_z_derivative_correction: zero,
                        self.initial_z_derivative_target: zero_z,
                        self.origin.reduced_z_derivative_correction: zero,
                        self.origin.z_derivative_target: zero_z
                    })

            if self.model_params["model_continuity_error_coeff"] != None:
                for i in range(self.sub_sequences_overlap):
                    if i not in self.std_vae_indices:
                        previous_dic.update({
                            self.vae_subsequence[i].reduced_var_correction: zero,
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
                # values shape = [nb_values, shape(val)]
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
                if self.z_continuity_error_coeff != None or self.model_params["model_continuity_error_coeff"] != None:
                    p = ()
                    index = self.sub_sequences_step

                    if self.z_continuity_error_coeff != None:

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

                    if self.model_params["model_continuity_error_coeff"] != None:
                        for i in range(index, self.sub_sequences_size):
                            if i not in self.std_vae_indices:
                                p += (self.vae_subsequence[i].var_to_correct,)

                    values = list(self.session.run(p, feed_dict=f_dict))
                    previous_dic = {}
                    offset = 0
                    if self.z_continuity_error_coeff != None:
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
                                        self.vae_subsequence[i].reduced_z_derivative_correction: zero,
                                        self.vae_subsequence[i].z_derivative_target: zero_z
                                    })

                            if index in z_derivative_indices:
                                # in case initial_z_derivative_target and z_derivative_target from the first VAE of subsequece aren't the same
                                previous_dic[self.initial_z_derivative_target] = values[offset]
                            else:
                                # all in all, if you hadn't set z_derivative value for the first VAE of subsequence, you hadn't set initial_z_derivative_target
                                previous_dic.update({
                                    self.reduced_initial_z_derivative_correction: zero,
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
                                    self.goal_vae.reduced_z_correction: zero,
                                    self.goal_vae.z_target: zero_z
                                })

                    if self.model_params["model_continuity_error_coeff"] != None:
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
                    self.annealing_schedule: annealing_schedule,
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

        return values_filled_subs


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

            if self.z_continuity_error_coeff != None and self.use_z_derivative:
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
            if self.z_continuity_error_coeff != None and self.use_z_derivative:
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

            variances = np.array(variances)
            # means shape = [sub_sequences_size, nb_sub_sequences]
            means = np.array(means)
            offset = 0
            if self.z_continuity_error_coeff != None and self.use_z_derivative:
                offset = -1
            global_cost = np.divide(np.sum(means, 0), offset + len(self.cost_per_vae))
        else:
            if self.separate_std_vaes:
                # train all of the standard VAEs first and retrieve their original costs
                means_goal, variances_goal, means_initial_z_derivative, variances_initial_z_derivative = train_all_std_vaes(means, variances)

            p = (self.cost, self.optimization,)
            for mean, var in zip(self.cost_per_vae, self.var_per_vae):
                p += (mean, var)

            # for vae in self.vae_subsequence:
            #     p += (self.vae_subsequence[i].reconstr_loss,)
            # values shape = [nb_sub_sequences, 2, shape(val)]
            values = self.get_values(p, Xs, annealing_schedule=annealing_schedule, fill=False, input_dicts=std_vae_dicts)
            values = np.array(values, dtype=np.float32)

            if True in np.isinf(values[:, 0]):
                print("INF cost !")
                self.crashed = True
            if True in np.isnan(values[:, 0]):
                print("NAN cost !")
                self.crashed = True

            variances = np.transpose(values[:, 3::2])
            # means shape = [sub_sequences_size, nb_sub_sequences]
            means = np.transpose(values[:, 2::2])
            offset = 0
            if self.z_continuity_error_coeff != None and self.use_z_derivative:
                offset = -1
            global_cost = np.divide(values[:, 0], offset + len(self.cost_per_vae))

        if np.array(means_goal).any():
            # at the end, add goal data to means and variances to always keep the same VAE order
            means = np.append(means, [means_goal], axis=0)
            variances = np.append(variances, [variances_goal], axis=0)

            if self.z_continuity_error_coeff != None and self.use_z_derivative:
                means = np.append(means, [means_initial_z_derivative], axis=0)
                variances = np.append(variances, [variances_initial_z_derivative], axis=0)

        # global_cost shape = [nb_sub_sequences], means and variances shape = [nb_sub_sequences, nb_vaes]
        return global_cost, np.transpose(means), np.transpose(variances)


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

        values = self.get_values(p, Xs, transform_with_all_vtsfe=transform_with_all_vtsfe)

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

        values = self.get_values(p, Xs, transform_with_all_vtsfe=transform_with_all_vtsfe, fill=fill)

        return values


    def init_session(self):
        if not self.restore or self.save_path == None or not os.path.isfile("./checkpoints/"+self.save_path+'.index'):
            # Initializing the tensor flow variables
            self.session.run(tf.global_variables_initializer())
        else:
            # Restore variables from disk
            tf.train.Saver().restore(self.session, "./checkpoints/"+self.save_path)
            print("\n---- VTSFE "+self.save_path+" RESTORED ----\n")


    def save_session(self, data_driver, data):
        # Save the variables to disk.
        if self.save_path != None:
            tf.train.Saver().save(self.session, "./checkpoints/"+self.save_path)
            data_driver.save_data("./training_errors/"+self.save_path, "errors", data)

            print("\n---- VTSFE SAVED IN FILE: ./checkpoints/"+self.save_path+" ----\n")
            print("\n---- TRAINING ERRORS SAVED IN FILE: ./training_errors/"+self.save_path+" ----\n")


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
            if self.restore:
                global_error, vae_errors, vae_variances = training["data_driver"].read_data("./training_errors/"+self.save_path, "errors")
            for epoch in range(training["nb_epochs"]):
                nb_batches = int(n_samples / training["batch_size"])
                remains_batch_size = n_samples % training["batch_size"]
                if remains_batch_size != 0:
                    nb_batches += 1

                if annealing_schedule_temp != None and annealing_schedule_temp != 0:
                    annealing_schedule = min(1., 1E-2 + epoch / annealing_schedule_temp)
                    # annealing_schedule = max(1., annealing_schedule_temp / (1E-2 + epoch))
                    # annealing_schedule = max(1., 1E-2 + epoch / annealing_schedule_temp)
                else:
                    annealing_schedule = 1.

                w = 1.
                nb_samples = 0
                avg_cost = 0.
                # 1 origin
                nb_vaes = 1
                if not self.sub_sequences_size-1 in self.std_vae_indices:
                    # 1 goal
                    nb_vaes += 1
                if self.z_continuity_error_coeff != None and self.use_z_derivative:
                    # + 1 fake VAE for the initial z derivative continuity constraint
                    nb_vaes += 1
                nb_vaes += len(self.std_vaes_like_origin) + self.sub_sequences_size - len(self.std_vae_indices)
                avg_errors = np.full(nb_vaes, 0.)
                intra_batch_variances = np.full(nb_vaes, 0.)
                inter_batch_variances = np.full(nb_vaes, 0.)
                batch_errors = []

                # Loop over all batches
                for i in range(nb_batches):
                    cost = 0.
                    batch = data[training["batch_size"]*i : training["batch_size"]*(i+1)]
                    # Fit training using batch data and average cost on all subsequences
                    # batch_cost shape = [nb_sub_sequences], means and variances shape = [nb_sub_sequences, nb_vaes]
                    batch_cost, batch_error_means, batch_error_variances = self.partial_fit(batch, annealing_schedule)

                    if self.crashed:
                        break

                    # averages over subsequences
                    bc_mean = np.mean(batch_cost, 0)
                    bem_mean = np.mean(batch_error_means, 0)
                    bev_mean = np.mean(batch_error_variances, 0)

                    batch_errors.append(bc_mean)

                    # Compute average loss
                    nb_samples += len(batch)
                    w = len(batch)/nb_samples
                    avg_cost = avg_cost*(1. - w) + w*bc_mean

                    # Compute average loss and variance on each VAE
                    for k in range(nb_vaes):
                        avg_errors[k] = avg_errors[k]*(1. - w) + w*bem_mean[k]
                        intra_batch_variances[k] = intra_batch_variances[k]*(1. - w) + w*bev_mean[k]

                if self.crashed:
                    break

                global_error.append(avg_cost)
                vae_errors.append(avg_errors)

                # Compute also inter-batch variance
                nb_samples = remains_batch_size
                if nb_samples == 0:
                    nb_samples = training["batch_size"]
                w = 1.
                # go through batches from the last one to the first one
                for b in range(nb_batches-1, -1, -1):
                    for k in range(nb_vaes):
                        inter_batch_variances[k] = inter_batch_variances[k]*(1. - w) + w*np.square(batch_errors[b] - avg_cost)
                    nb_samples += training["batch_size"]
                    w = training["batch_size"]/nb_samples

                # Compute the total variance per VAE
                total_variances = np.add(intra_batch_variances, inter_batch_variances)
                vae_variances.append(total_variances)

                # Display logs per epoch step
                if epoch % training["display_step"] == 0 or epoch == training["nb_epochs"]-1:
                    print("Epoch:", '%04d' % (epoch),
                          " ----------> Average cost =", "{:.9f}".format(avg_cost))

                if epoch > 0 and epoch % training["checkpoint_step"] == 0:
                    self.save_session(training["data_driver"], (global_error, vae_errors, vae_variances))

            end = datetime.now()
            print("\n------- Training end: {} -------\n".format(end.isoformat()[11:]))
            print("Elapsed = "+str((end-begin).total_seconds())+" seconds\n")

            self.save_session(training["data_driver"], (global_error, vae_errors, vae_variances))
            return np.array(global_error), np.array(vae_errors), np.array(vae_variances)


    def plot_error(self, global_error, errors, variances):
        nb_vaes = len(errors[0])
        nb_epochs = len(global_error)

        variables = []
        labels = ["Global error", "Error mean", "Error +/- sigma"]
        linestyles = ["dashed", "-", "-"]
        colors = cm.rainbow(np.linspace(0, 1, len(labels)))
        vaes_labels = (
            ["Origin VAE"]
            + ["Standard VAE"]*len(self.std_vaes_like_origin)
            + ["Model VAE"]*(self.sub_sequences_size-len(self.std_vae_indices))
            + ["Goal VAE"]
        )

        if self.z_continuity_error_coeff != None and self.use_z_derivative:
            vaes_labels += ["z derivative continuity"]

        def set_information(raw_variables):
            for v, var in enumerate(raw_variables):
                variables.append({
                    "values": np.array(var),
                    "label": labels[v],
                    "linestyle": linestyles[v],
                    "color": colors[v]
                })

        set_information([global_error, errors])

        sigmas = np.sqrt(variances)
        sigma_sup = np.add(errors, sigmas)
        sigma_inf = np.subtract(errors, sigmas)

        column_size = int(np.sqrt(nb_vaes))
        plots_mod = nb_vaes % column_size
        row_size = int(nb_vaes / column_size)
        if plots_mod != 0:
            row_size += 1

        fig, axes = plt.subplots(column_size, row_size, sharex=True, sharey=True, figsize=(50,100))
        fig.canvas.set_window_title("Error through epochs per VAE")

        if nb_vaes == 1:
            axes = np.array([axes])
        else:
            axes = axes.reshape([-1])

        plots = []
        def plot_variable(ax=None, indices=slice(0, None), values=[], linestyle="-", label=None, color=None):
            ax.plot(
                    range(nb_epochs),
                    values[indices],
                    c=color,
                    label=label,
                    linestyle=linestyle
                )
            if label != None:
                ax.legend(bbox_to_anchor=(-0.2, 2), loc=1, borderaxespad=0.)

        for i in range(nb_vaes):
            ax = axes[i]
            ax.grid()
            ax.margins(0.)
            ax.set_title(vaes_labels[i])
            if i == 0:
                label = labels[-1]
            else:
                label = None
            # plot a colored surface between mu-sigma and mu+sigma
            ax.fill_between(range(nb_epochs), sigma_inf[:, i], sigma_sup[:, i], alpha=0.5, label=labels[-1], facecolor=colors[-1])
            # plot global and local means
            variables[1]["indices"] = (slice(0, None), i)
            for k, var in enumerate(variables):
                if not self.use_z_derivative or self.z_continuity_error_coeff == None or i != nb_vaes-1 or k != 0:
                    # if it isn't the z derivative continuity error plot and the global_error variable
                    var.update({
                        "ax": ax
                    })
                    plot_variable(**var)
                    var["label"] = None
        plt.show()


    def get_reconstruction_squarred_error(self, data, transform_with_all_vtsfe=True):
        x_samples = data
        # x_reconstr shape = [nb_sub_sequences, sub_sequences_size, nb_samples, n_input]
        x_reconstr = np.array(self.reconstruct(x_samples, transform_with_all_vtsfe=transform_with_all_vtsfe, fill=False), dtype=np.float32)
        if transform_with_all_vtsfe:
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
            # transpose x_seq to shape [nb_samples, nb_frames, n_input]
            x_seq = np.transpose(x_seq, [1, 0, 2])
        else:
            x_seq = np.reshape(x_reconstr, [-1, self.nb_frames, self.vae_architecture["n_input"]])

        return np.square(np.subtract(x_samples, x_seq))


    def show_latent_space(self, data_driver, latent_data, sample_indices, title, displayed_movs=[], nb_samples_per_mov=1, show_frames=False):
        """
            Latent space representation
        """
        labels = data_driver.data_labels

        # don't display more movements than there are
        if nb_samples_per_mov > len(sample_indices):
            display_per_mov = len(sample_indices)
        else:
            display_per_mov = nb_samples_per_mov

        # translate observations in latent space
        # latent_data shape = [nb_sub_sequences, nb_frames, nb_samples, n_z]
        z_mu = np.array(latent_data)

        print("input space dimensions = "+str(self.vae_architecture["n_input"]))
        print("latent space dimensions = "+str(self.vae_architecture["n_z"]))

        # indices_per_mov_type = []
        # for mov in displayed_movs:
        #     mov_index = data_driver.mov_indices[mov]
        #     indices_per_mov_type.append(np.where(labels == mov_index))

        nb_mov_types = len(displayed_movs)

        colors = cm.rainbow(np.linspace(0, 1, nb_mov_types))

        # if either latent space isn't in 3-D or you want to see the time axis
        if self.vae_architecture["n_z"] != 3 or show_frames:
            if show_frames:
                # if you want to see evolution through time,
                # plot the necessary number of 3-D plots with an axis for time, and 2 for latent space

                def plot_sample(ax, s, i, flat_index, x, y, marker=None, markevery=None, label=None):
                    ax.plot(
                            range(self.nb_frames),
                            z_mu[s, :, flat_index, x],
                            z_mu[s, :, flat_index, y],
                            c=colors[i],
                            label=label,
                            marker=marker,
                            markevery=markevery
                        )

                def add_markers(ax, s, i, flat_index, x, y):
                    # mark start of sequence
                    ax.scatter(
                            0,
                            z_mu[s, 0, flat_index, x],
                            z_mu[s, 0, flat_index, y],
                            c=colors[i],
                            marker='^',
                            s=320
                        )
                    # mark end of sequence
                    ax.scatter(
                            self.nb_frames-1,
                            z_mu[s, self.nb_frames-1, flat_index, x],
                            z_mu[s, self.nb_frames-1, flat_index, y],
                            c=colors[i],
                            marker='v',
                            s=320
                        )

                # turn on interactive mode
                plt.ion()

                # plot every permutation of 2 dimensions of latent space
                for x in range(self.vae_architecture["n_z"]):
                    for y in range(x+1, self.vae_architecture["n_z"]):
                        fig = plt.figure(figsize=(50,100))
                        fig.canvas.set_window_title("Latent space - "+title)
                        ax = fig.gca(projection='3d')
                        ax.set_title("Axes = ["+str(x)+", "+str(y)+"]")

                        # plot a path through time per sample, a color per movement type
                        for i, mov in enumerate(displayed_movs):
                            for j in range(display_per_mov):
                                flat_index = data_driver.mov_indices[mov] + sample_indices[j]
                                for s in range(len(z_mu)):
                                    # add only one instance of the same label
                                    if j == 0 and s == 0:
                                        plot_sample(ax, s, i, flat_index, x, y, marker='o', markevery=slice(1, self.nb_frames-1, 1), label=labels[flat_index])
                                    else:
                                        plot_sample(ax, s, i, flat_index, x, y, marker='o', markevery=slice(1, self.nb_frames-1, 1))

                                    add_markers(ax, s, i, flat_index, x, y)

                        plt.legend(bbox_to_anchor=(0.9, 0.9), loc=3, borderaxespad=0.)
                        plt.show()
                        _ = input("Press [enter] to continue.") # wait for input from the user
                        plt.close()    # close the figure to show the next one.

                # turn off interactive mode
                plt.ioff()

            else:
                # if you don't want to see evolution through time,
                # plot the necessary number of 2-D plots with 2 axes for latent space

                def plot_sample(ax, s, i, flat_index, x, y, marker=None, markevery=None, label=None):
                    ax.plot(
                            z_mu[s, :, flat_index, x],
                            z_mu[s, :, flat_index, y],
                            c=colors[i],
                            label=label,
                            marker=marker,
                            markevery=markevery
                        )

                def add_markers(ax, s, i, flat_index, x, y):
                    # mark start of sequence
                    ax.scatter(
                            z_mu[s, 0, flat_index, x],
                            z_mu[s, 0, flat_index, y],
                            c=colors[i],
                            marker='^',
                            s=320
                        )
                    # mark end of sequence
                    ax.scatter(
                            z_mu[s, self.nb_frames-1, flat_index, x],
                            z_mu[s, self.nb_frames-1, flat_index, y],
                            c=colors[i],
                            marker='v',
                            s=320
                        )

                # plot every permutation of 2 dimensions of latent space in the same figure
                # wrap these configurations in a 1-D array named 'plots' to use with the 1-D array of axes
                plots = []
                for x in range(self.vae_architecture["n_z"]):
                    for y in range(x+1, self.vae_architecture["n_z"]):
                        plots.append({
                            "x": x,
                            "y": y
                        })

                column_size = int(np.sqrt(len(plots)))
                plots_mod = len(plots) % column_size
                row_size = int(len(plots) / column_size)
                if plots_mod != 0:
                    row_size += 1

                fig, axes = plt.subplots(column_size, row_size, sharex=True, sharey=True, figsize=(50,100))
                fig.canvas.set_window_title("Latent space - "+title)

                if len(plots) == 1:
                    axes = np.array([axes])
                else:
                    axes = axes.reshape([-1])

                for p, plot in enumerate(plots):
                    ax = axes[p]
                    ax.set_title("Axes = ["+str(plot["x"])+", "+str(plot["y"])+"]")
                    # plot a path per sample, a color per movement type
                    for i, mov in enumerate(displayed_movs):
                        for j in range(display_per_mov):
                            flat_index = data_driver.mov_indices[mov] + sample_indices[j]
                            for s in range(len(z_mu)):
                                # add only one instance of the same label
                                if j == 0 and s == 0:
                                    plot_sample(ax, s, i, flat_index, plot["x"], plot["y"], marker='o', markevery=slice(1, self.nb_frames-1, 1), label=labels[flat_index])
                                else:
                                    plot_sample(ax, s, i, flat_index, plot["x"], plot["y"], marker='o', markevery=slice(1, self.nb_frames-1, 1))

                                add_markers(ax, s, i, flat_index, plot["x"], plot["y"])

                plt.legend(bbox_to_anchor=(0.9, 0.9), loc=3, borderaxespad=0.)
                plt.show()

        else:
            # if latent space is in 3-D and you don't want to see the time dimension,
            # just plot all axes in one 3-D plot

            def plot_sample(ax, s, i, flat_index, marker=None, markevery=None, label=None):
                ax.plot(
                        z_mu[s, :, flat_index, 0],
                        z_mu[s, :, flat_index, 1],
                        z_mu[s, :, flat_index, 2],
                        c=colors[i],
                        label=label,
                        marker=marker,
                        markevery=markevery
                    )

            def add_markers(ax, s, i, flat_index):
                # mark start of sequence
                ax.scatter(
                        z_mu[s, 0, flat_index, 0],
                        z_mu[s, 0, flat_index, 1],
                        z_mu[s, 0, flat_index, 2],
                        c=colors[i],
                        marker='^',
                        s=320
                    )
                # mark end of sequence
                ax.scatter(
                        z_mu[s, self.nb_frames-1, flat_index, 0],
                        z_mu[s, self.nb_frames-1, flat_index, 1],
                        z_mu[s, self.nb_frames-1, flat_index, 2],
                        c=colors[i],
                        marker='v',
                        s=320
                    )

            fig = plt.figure(figsize=(50,100))
            fig.canvas.set_window_title("Latent space - "+title)
            ax = fig.gca(projection='3d')
            # plot a path per sample, a color per movement type
            for i, mov in enumerate(displayed_movs):
                for j in range(display_per_mov):
                    flat_index = data_driver.mov_indices[mov] + sample_indices[j]
                    for s in range(len(z_mu)):
                        # add only one instance of the same label
                        if j == 0 and s == 0:
                            plot_sample(ax, s, i, flat_index, marker='o', markevery=slice(1, self.nb_frames-1, 1), label=labels[flat_index])
                        else:
                            plot_sample(ax, s, i, flat_index, marker='o', markevery=slice(1, self.nb_frames-1, 1))

                        add_markers(ax, s, i, flat_index)

            plt.legend(bbox_to_anchor=(0.8, 0.7), loc=2, borderaxespad=0.)
            plt.show()


    def show_data(self, data_driver, reconstr_datasets, reconstr_datasets_names, sample_indices, x_samples, nb_samples_per_mov=1, displayed_movs=None, plot_3D=False, window_size=10, time_step=5, body_lines=True, only_hard_joints=True, transform_with_all_vtsfe=True):
        """
            Data space representation.
            You can optionally plot reconstructed data as well at the same time.
        """
        labels = data_driver.data_labels

        # don't display more movements than there are
        if nb_samples_per_mov > len(sample_indices):
            display_per_mov = len(sample_indices)
        else:
            display_per_mov = nb_samples_per_mov

        if transform_with_all_vtsfe:
            nb_sub_sequences = self.nb_sub_sequences
        else:
            nb_sub_sequences = 1

        nb_colors = display_per_mov
        if len(reconstr_datasets) > 0:
            nb_colors *= len(reconstr_datasets)
        colors = cm.rainbow(np.linspace(0, 1, nb_colors))

        if plot_3D:
            # to avoid modifying x_samples
            data = np.copy(x_samples)
            # x_samples shape = [nb_samples, nb_frames, segment_count, 3 (coordinates)]
            data = data.reshape([len(data_driver.mov_types)*data_driver.nb_samples_per_mov, self.nb_frames, -1, 3])

            data_reconstr = []
            for reco in reconstr_datasets:
                data_reconstr.append(reco.reshape([nb_sub_sequences, len(data_driver.mov_types)*data_driver.nb_samples_per_mov, self.nb_frames, -1, 3]))

            segment_count = len(data[0, 0])

            cs = []

            if not body_lines:
                # color every segment point, a color per sample
                for i in range(display_per_mov):
                    for j in range(segment_count):
                        cs.append(colors[i])

            # plots = []
            # plot_recs = []

            body_lines_indices = [[0, 7], [7, 11], [11, 15], [15, 19], [19, 23]]
            additional_lines = [[15, 0, 19], [7, 11]]
            nb_body_lines = len(body_lines_indices)
            nb_additional_lines = len(additional_lines)

            if body_lines:
                def plot_body_lines(plots, j, k, data):
                    for i in range(nb_body_lines):
                        line_length = body_lines_indices[i][1] - body_lines_indices[i][0]
                        # NOTE: there is no .set_data() for 3 dim data...
                        # plot 2D
                        plots[i].set_data(
                            data[
                                data_driver.mov_indices[mov] + j,
                                k,
                                body_lines_indices[i][0] : body_lines_indices[i][1],
                                0
                            ],
                            data[
                                data_driver.mov_indices[mov] + j,
                                k,
                                body_lines_indices[i][0] : body_lines_indices[i][1],
                                1
                            ]
                        )
                        # plot the 3rd dimension
                        plots[i].set_3d_properties(
                            data[
                                data_driver.mov_indices[mov] + j,
                                k,
                                body_lines_indices[i][0] : body_lines_indices[i][1],
                                2
                            ]
                        )

                    for i in range(nb_additional_lines):
                        # additional_lines_data shape = [display_per_mov, nb_additional_lines, 3, line_length, nb_frames]

                        # plot 2D
                        plots[nb_body_lines+i].set_data(
                            data[
                                data_driver.mov_indices[mov] + j,
                                k,
                                additional_lines[i],
                                0
                            ],
                            data[
                                data_driver.mov_indices[mov] + j,
                                k,
                                additional_lines[i],
                                1
                            ]
                        )
                        # plot the 3rd dimension
                        plots[nb_body_lines+i].set_3d_properties(
                            data[
                                data_driver.mov_indices[mov] + j,
                                k,
                                additional_lines[i],
                                2
                            ]
                        )


                def animate(k):
                    index = 0
                    step = nb_body_lines + nb_additional_lines
                    for j in range(display_per_mov):
                        next_index = index + step
                        plot_body_lines(plots[index : next_index], sample_indices[j], k, data)

                        for r,reco in enumerate(data_reconstr):
                            for sub in range(nb_sub_sequences):
                                plot_body_lines(plot_recs[r*nb_sub_sequences + sub][index : next_index], r*nb_sub_sequences + sample_indices[j], k, reco[sub])
                        index = next_index
                    title.set_text("Time = {}".format(k))
                    ax.view_init(30, -150 + 0.7 * k)
            else:
                def animate(k):
                    indices = [data_driver.mov_indices[mov] + j for j in sample_indices]
                    plots[0]._offsets3d = (
                        data[indices, k, :, 0].reshape([segment_count*display_per_mov]),
                        data[indices, k, :, 1].reshape([segment_count*display_per_mov]),
                        data[indices, k, :, 2].reshape([segment_count*display_per_mov])
                    )
                    for r,reco in enumerate(data_reconstr):
                        for sub in range(nb_sub_sequences):
                            plot_recs[r*nb_sub_sequences + sub]._offsets3d = (
                                reco[sub, indices, k, :, 0].reshape([segment_count*display_per_mov]),
                                reco[sub, indices, k, :, 1].reshape([segment_count*display_per_mov]),
                                reco[sub, indices, k, :, 2].reshape([segment_count*display_per_mov])
                            )
                    title.set_text("Time = {}".format(k))
                    ax.view_init(30, -150 + 0.7 * k)

            if displayed_movs != None:
                # turn on interactive mode
                plt.ion()
                # scatter all movements in displayed_movs, a color per movement sample
                for i, mov in enumerate(displayed_movs):
                    plots = []
                    plot_recs = []
                    fig = plt.figure(figsize=(8, 6))
                    fig.canvas.set_window_title("Input data space - "+mov)
                    ax = fig.gca(projection='3d')
                    box_s = 1
                    ax.set_xlim3d(-box_s, box_s)
                    ax.set_ylim3d(-box_s, box_s)
                    ax.set_zlim3d(-box_s, box_s)
                    # set point-of-view: specified by (altitude degrees, azimuth degrees)
                    ax.view_init(30, -150)
                    title = ax.set_title("Time = 0")

                    if body_lines:
                        for j in range(display_per_mov):
                            # plot the nb_body_lines+nb_additional_lines lines of body segments
                            for k in range(nb_body_lines + nb_additional_lines):
                                # plots shape = [display_per_mov*(nb_body_lines + nb_additional_lines)]
                                plots.append(ax.plot([], [], [], c=colors[j], marker='o')[0])

                        for r in range(len(reconstr_datasets)):
                            label = reconstr_datasets_names[r]
                            for sub in range(nb_sub_sequences):
                                plts = []
                                for j in range(display_per_mov):
                                    # plot the nb_body_lines + nb_additional_lines lines of body reconstructed segments
                                    for k in range(nb_body_lines + nb_additional_lines):
                                        if j != 0 or k != 0 or sub != 0:
                                            label = None
                                        plts.append(ax.plot([], [], [], label=label, c=colors[r*nb_sub_sequences + j], marker='D', linestyle='dashed')[0])
                                # plot_recs shape = [nb_reconstr_data*nb_sub_sequences, display_per_mov*(nb_body_lines + nb_additional_lines)]
                                plot_recs.append(plts)
                    else:
                        # just scatter every point
                        plots.append(ax.scatter([], [], [], c=cs))
                        for r in range(len(reconstr_datasets)):
                            label = reconstr_datasets_names[r]
                            plot_recs.append(ax.scatter([], [], [], label=label, c=cs, marker='D'))

                    # call the animator.  blit=True means only re-draw the parts that have changed.
                    anim = animation.FuncAnimation(fig, animate, frames=self.nb_frames, interval=time_step, blit=False)
                    # Save as mp4. This requires mplayer or ffmpeg to be installed
                    if self.save_path == None:
                        s_path = "default"
                    else:
                        s_path = self.save_path
                    plt.legend(bbox_to_anchor=(0.75, 1), loc=3, borderaxespad=0.)
                    anim.save("./animations/input_data_space-"+s_path+"-"+mov+".mp4", fps=15, extra_args=['-vcodec', 'libx264'])
                    plt.show()
                    _ = input("Press [enter] to continue.") # wait for input from the user
                    plt.close()    # close the figure to show the next one.

                # turn off interactive mode
                plt.ioff()
        else:
            # if it's not a 3D plot

            # turn on interactive mode
            plt.ion()

            if only_hard_joints:
                source_list = data_driver.hard_dimensions
                sources_count = len(source_list)
            else:
                sources_count = len(x_samples[0, 0])
                source_list = range(sources_count)

            column_size = int(np.sqrt(sources_count))
            plots_mod = sources_count % column_size
            row_size = int(sources_count / column_size)
            if plots_mod != 0:
                row_size += 1

            # add a plot to animate
            def add_plot(plots, ax, j, linestyle="-", label=None):
                # plots shape = [sources_count*displayed_movs*display_per_mov*(1+nb_sub_sequences)]
                # or
                # plots shape = [sources_count*displayed_movs*display_per_mov]
                plots.append(
                    ax.plot(
                        [],
                        [],
                        c=colors[j],
                        linestyle=linestyle,
                        label=label
                    )[0]
                )

                if label != None:
                    ax.legend(bbox_to_anchor=(2, 1.6), loc=1, borderaxespad=0.)
                return plots

            if len(data_driver.data_types) == 1:
                dim_names_available = True
            else:
                dim_names_available = False

            for i, mov in enumerate(displayed_movs):
                fig, axes = plt.subplots(column_size, row_size, sharex=True, sharey=True, figsize=(50,100))
                fig.canvas.set_window_title("Input data space of "+mov)
                if sources_count == 1:
                    axes = np.array([axes])
                else:
                    axes = axes.reshape([-1])
                plots = []
                # for each source type
                # wrap all plots in a 1-D array named 'plots' for the animation function
                for s,source in enumerate(source_list):
                    ax = axes[s]
                    ax.set_xlim(0, window_size-1)
                    ax.set_ylim(-1, 1)
                    ax.grid()
                    if dim_names_available:
                        ax.set_title(data_driver.dim_names[source])
                    # plot all movements types in same plot, each with a different color
                    # plot display_per_mov samples of the same movement type with the same color
                    for j in range(display_per_mov):
                        flat_index = data_driver.mov_indices[mov] + sample_indices[j]

                        if j == 0 and s == 0:
                            # input data plot
                            plots = add_plot(plots, ax, j, label=labels[flat_index])

                            for r,reco in enumerate(reconstr_datasets):
                                for sub in range(nb_sub_sequences):
                                    flat_r_index = r*nb_sub_sequences + j
                                    if sub == 0:
                                        # reconstructed data plot
                                        plots = add_plot(plots, ax, flat_r_index, label=reconstr_datasets_names[r], linestyle="dashed")
                                    else:
                                        # reconstructed data plot
                                        plots = add_plot(plots, ax, flat_r_index, linestyle="dashed")
                        else:
                            # input data plot
                            plots = add_plot(plots, ax, j)

                            for r,reco in enumerate(reconstr_datasets):
                                for sub in range(nb_sub_sequences):
                                    # reconstructed data plot
                                    plots = add_plot(plots, ax, r*nb_sub_sequences + j, linestyle="dashed")

                # animation function.  This is called sequentially
                def animate(t):
                    jump = nb_sub_sequences*len(reconstr_datasets) + 1

                    for s,source in enumerate(source_list):
                        ax = axes[s]
                        ax.set_xlim(t, t + window_size - 1)
                        mov_count = s*display_per_mov*jump
                        for j in range(display_per_mov):
                            flat_index = data_driver.mov_indices[mov] + sample_indices[j]
                            flat_index_plots = mov_count + jump*j

                            # plot input data
                            plots[flat_index_plots].set_data(
                                range(t, t + window_size +1 ),
                                x_samples[flat_index, t: t + window_size +1, source]
                            )
                            flat_index_plots += 1
                            for r,reco in enumerate(reconstr_datasets):
                                flat_r_index = r*nb_sub_sequences
                                for sub in range(nb_sub_sequences):
                                    # plot reconstructed data
                                    plots[flat_index_plots + flat_r_index + sub].set_data(
                                        range(t, t + window_size +1 ),
                                        reco[sub, flat_index, t: t + window_size +1, source]
                                    )
                    return plots

                # call the animator.  blit=True means only re-draw the parts that have changed.
                anim = animation.FuncAnimation(fig, animate, frames=self.nb_frames-window_size, interval=time_step, blit=False)
                plt.legend(bbox_to_anchor=(0.75, 1), loc=3, borderaxespad=0.)

                # Save as mp4. This requires mplayer or ffmpeg to be installed
                if self.save_path == None:
                    s_path = "default"
                else:
                    s_path = self.save_path
                anim.save("./animations/input_data_space-"+s_path+"-"+mov+".mp4", fps=15, extra_args=['-vcodec', 'libx264'])
                plt.show()
                _ = input("Press [enter] to continue.") # wait for input from the user
                plt.close()    # close the figure to show the next one.

            # turn off interactive mode
            plt.ioff()

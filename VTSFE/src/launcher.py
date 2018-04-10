# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pdb

from mpl_toolkits.mplot3d import Axes3D

from .data_driver import Data_driver
from .vtsfe import VTSFE


class Launcher():

    def __init__(self, config, lrs, restore_path=None, save_path=None):
        # For test purposes : initialization always starts with the same random values
        np.random.seed(0)
        tf.set_random_seed(0)
        #np.random.seed(None)
        #tf.set_random_seed(None)

        self.lrs = lrs
        self.vtsfe = None
        self.x_reconstr = None
        self.config = config
        self.restore_path = restore_path
        self.save_path = save_path
        self.config.DATA_PARAMS.update({
            "normalization_as_3d": self.config.DATA_PARAMS["as_3D"],
            "relative_movement": self.config.DATA_PARAMS["as_3D"],
            "use_center_of_mass": self.config.DATA_PARAMS["as_3D"]
        })
        self.data_driver = Data_driver(self.config.DATA_PARAMS)
        self.data_driver.parse()

        # data shape = [nb_mov*nb_samples_per_mov, nb_frames, dim(values)] = [nb_samples, nb_frames, dim(values)]
        self.config.DATA_VISUALIZATION = {
            "data_driver": self.data_driver,
            "displayed_movs": self.data_driver.mov_types,
            "transform_with_all_vtsfe": False,
            "average_reconstruction": False
        }
        self.config.TRAINING["data_driver"] = self.data_driver

        data_dim = self.data_driver.input_dim # dimensionality of observations
        self.config.VTSFE_PARAMS["vae_architecture"]["n_input"] = data_dim
        self.config.VTSFE_PARAMS["vae_architecture"]["n_output"] = data_dim
        self.config.VTSFE_PARAMS["nb_frames"] = self.data_driver.nb_frames


    def show_data(self, sample_indices=None, only_hard_joints=True):
        self.init_vtsfe()

        if sample_indices is None:
            s_indices = range(self.data_driver.nb_samples_per_mov)
        else:
            s_indices = sample_indices

        x_samples = self.data_driver.get_whole_data_set(shuffle_dataset=False)
        self.config.DATA_VISUALIZATION["reconstr_datasets"] = []
        self.config.DATA_VISUALIZATION["reconstr_datasets_names"] = []
        self.config.DATA_VISUALIZATION["x_samples"] = x_samples
        self.config.DATA_VISUALIZATION["sample_indices"] = s_indices
        self.config.DATA_VISUALIZATION["only_hard_joints"] = only_hard_joints
        self.vtsfe.show_data(
            self.config.DATA_VISUALIZATION
        )

    def show_data_ori(self, sample_indices=None):
        self.init_vtsfe()
        if sample_indices is None:
            s_indices = range(self.data_driver.nb_samples_per_mov)
        else:
            s_indices = sample_indices
        x_samples = self.data_driver.get_whole_data_set(shuffle_dataset=False)
        pdb.set_trace()

        self.config.DATA_VISUALIZATION["reconstr_datasets"] = []
        self.config.DATA_VISUALIZATION["reconstr_datasets_names"] = []
        self.config.DATA_VISUALIZATION["x_samples"] = x_samples
        self.config.DATA_VISUALIZATION["sample_indices"] = s_indices
        self.vtsfe.show_data(self.config.DATA_VISUALIZATION)

    def train(self, data, show_error=True, resume=False, save_path="default"):
        self.destroy_vtsfe()
        self.vtsfe = VTSFE(
            self.config.VTSFE_PARAMS,
            training_mode=True,
            restore=resume,
            save_path=save_path
        )
        e = self.vtsfe.train(data, self.config.TRAINING)
        if show_error:
            self.plot_error(e)


    def plot_error(self, errors):
        self.vtsfe.plot_error(errors)


    def plot_error(self, path):
        self.init_vtsfe()
        errors = self.data_driver.read_data("./training_errors/"+path, "errors")
        self.vtsfe.plot_error(errors)


    def get_reconstruction_squarred_error(self, data, reconstruction, transform_with_all_vtsfe=True):
        self.init_vtsfe()
        return self.vtsfe.get_reconstruction_squarred_error(data, reconstruction, transform_with_all_vtsfe=transform_with_all_vtsfe)


    def plot_variance_histogram(self, data):
        nb_samples_per_mov = int(data.shape[0]/len(self.data_driver.mov_types))
        fig = plt.figure(figsize=(20, 10))
        ax = fig.gca(projection='3d')

        se = self.get_reconstruction_squarred_error(data)
        mse, vse = np.mean(se, axis=1), np.var(se, axis=1)

        print("Total MSE = "+str(np.mean(mse)))
        print("Total VSE sum = "+str(np.sum(vse)))

        nb_samples = len(self.data_driver.mov_types)*nb_samples_per_mov
        nb_segments = self.data_driver.input_dim
        nb_points = nb_samples*nb_segments
        xpos = []
        ypos = []
        dz = []
        for i in range(len(self.data_driver.mov_types)):
            for j in range(nb_samples_per_mov):
                x = i*nb_samples_per_mov + j
                for y in range(nb_segments):
                    xpos.append(x)
                    ypos.append(y)
                    dz.append(vse[x, y])

        zpos = np.zeros(nb_points)
        dx = np.ones(nb_points)
        dy = np.ones(nb_points)
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz)
        plt.show()


    def plot_mse(self, compare_to_other_models=True, sample_indices=None):
        self.init_vtsfe() #initialise VTSFE avec mode non training, restore, et un restore_path
        x_samples = self.data_driver.get_whole_data_set(shuffle_dataset=False)
        self.init_x_reconstr()
        if sample_indices is None:
            s_indices = range(self.data_driver.nb_samples_per_mov)
        else:
            s_indices = sample_indices
        # se = self.se[:,:6,:,:]
        se = self.se
        #pdb.set_trace()
        se_dataset = [se]
        se_names = [self.save_path]
        if compare_to_other_models:
            for lr in self.lrs:
                if lr.x_reconstr is not None:
                    # se = lr.se[:,:6,:,:]
                    se = lr.se
                    se_dataset.append(se)
                    se_names.append(lr.save_path)
        se_dataset = np.array(se_dataset)
        DATA_VISUALIZATION = {
            "se_dataset": se_dataset,
            "se_names": se_names,
            "displayed_movs": self.data_driver.mov_types,
            "s_indices": s_indices
        }
        self.vtsfe.plot_mse(
            DATA_VISUALIZATION
        )


    def init_vtsfe(self):
        if self.vtsfe is None:
            if self.restore_path is not None:
                restore = True
            else:
                restore = False

            self.vtsfe = VTSFE(
                self.config.VTSFE_PARAMS,
                training_mode=False,
                restore=restore,
                save_path=self.restore_path
            )
            
    #def init_x_reconstr_zs(self, zs):
        #if self.x_reconstr is None:
            #self.init_vtsfe()

            #x_samples = self.data_driver.get_whole_data_set(shuffle_dataset=False)
            #if self.config.DATA_VISUALIZATION["transform_with_all_vtsfe"] and self.config.DATA_VISUALIZATION["average_reconstruction"]:
                ## x_reconstr shape = [1, nb_frames, nb_samples, n_input]
                #x_reconstr = np.array(self.vtsfe.reconstruct_fromLS(zs, transform_with_all_vtsfe=False, fill=False))
            #else:
                ## x_reconstr shape = [nb_sub_sequences, sub_sequences_size, nb_samples, n_input] or [1, nb_frames, nb_samples, n_input]
                #x_reconstr = np.array(self.vtsfe.reconstruct_fromLS(zs, transform_with_all_vtsfe=self.config.DATA_VISUALIZATION["transform_with_all_vtsfe"], fill=False))

            #if self.config.DATA_VISUALIZATION["transform_with_all_vtsfe"] and not self.config.DATA_VISUALIZATION["average_reconstruction"]:
                ## x_reconstr shape = [nb_sub_sequences, nb_samples, nb_frames, n_input]
                    #self.x_reconstr = np.array(self.vtsfe.reconstruct_fromLS(zs, transform_with_all_vtsfe=True))
            #else:
                #if self.config.DATA_VISUALIZATION["transform_with_all_vtsfe"]:
                    #self.config.DATA_VISUALIZATION["transform_with_all_vtsfe"] = False
                    ## x_reconstr shape [1, nb_frames, nb_samples, n_input]
                    #x_reconstr = np.reshape(self.vtsfe.from_subsequences_to_sequence(x_samples, x_reconstr), [1, self.vtsfe.nb_frames, -1, self.vtsfe.vae_architecture["n_input"]])
                #self.x_reconstr = x_reconstr
            ## se shape = [nb_samples, nb_frames, n_input]
            
            #se = self.get_reconstruction_squarred_error(x_samples, x_reconstr, transform_with_all_vtsfe=self.config.DATA_VISUALIZATION["transform_with_all_vtsfe"])
            ## x_reconstr shape = [nb_sub_sequences, nb_samples, nb_frames, n_input]
            #self.x_reconstr = np.transpose(self.x_reconstr, [0, 2, 1, 3])
            ## reshaped se shape = [nb_mov_types, nb_samples_per_mov, nb_frames, n_input]
            ##7*10*70*66 à partir de x_samples = 70*70*66
            #self.se = np.reshape(se, [len(self.data_driver.mov_types), -1, self.vtsfe.nb_frames, self.vtsfe.vae_architecture["n_input"]])


    def init_x_reconstr(self):
        if self.x_reconstr is None:
            self.init_vtsfe()
            # x_samples = self.data_driver.get_whole_data_set(shuffle_dataset=False)
            # # x_reconstr shape = [nb_sub_sequences, nb_samples, nb_frames, n_input]
            # self.x_reconstr = np.transpose(self.vtsfe.reconstruct(x_samples, transform_with_all_vtsfe=self.config.DATA_VISUALIZATION["transform_with_all_vtsfe"]), [0, 2, 1, 3])
            #
            # # x_reconstr shape = [nb_sub_sequences, sub_sequences_size, nb_samples, n_input]
            # x_reconstr = np.array(self.vtsfe.reconstruct(x_samples, transform_with_all_vtsfe=self.config.DATA_VISUALIZATION["transform_with_all_vtsfe"], fill=False))
            # # se shape = [nb_samples, nb_frames, n_input]
            # se = self.get_reconstruction_squarred_error(x_samples, x_reconstr, transform_with_all_vtsfe=self.config.DATA_VISUALIZATION["transform_with_all_vtsfe"])
            # # reshaped se shape = [nb_mov_types, nb_samples_per_mov, nb_frames, n_input]
            # self.se = np.reshape(se, [len(self.data_driver.mov_types), -1, self.vtsfe.nb_frames, self.vtsfe.vae_architecture["n_input"]])

            x_samples = self.data_driver.get_whole_data_set(shuffle_dataset=False)
            if self.config.DATA_VISUALIZATION["transform_with_all_vtsfe"] and self.config.DATA_VISUALIZATION["average_reconstruction"]:
                # x_reconstr shape = [1, nb_frames, nb_samples, n_input]
                x_reconstr = np.array(self.vtsfe.reconstruct(x_samples, transform_with_all_vtsfe=False, fill=False))
            else:
                # x_reconstr shape = [nb_sub_sequences, sub_sequences_size, nb_samples, n_input] or [1, nb_frames, nb_samples, n_input]
                x_reconstr = np.array(self.vtsfe.reconstruct(x_samples, transform_with_all_vtsfe=self.config.DATA_VISUALIZATION["transform_with_all_vtsfe"], fill=False))

            if self.config.DATA_VISUALIZATION["transform_with_all_vtsfe"] and not self.config.DATA_VISUALIZATION["average_reconstruction"]:
                # x_reconstr shape = [nb_sub_sequences, nb_samples, nb_frames, n_input]
                    self.x_reconstr = np.array(self.vtsfe.reconstruct(x_samples, transform_with_all_vtsfe=True))
            else:
                if self.config.DATA_VISUALIZATION["transform_with_all_vtsfe"]:
                    self.config.DATA_VISUALIZATION["transform_with_all_vtsfe"] = False
                    # x_reconstr shape [1, nb_frames, nb_samples, n_input]
                    x_reconstr = np.reshape(self.vtsfe.from_subsequences_to_sequence(x_samples, x_reconstr), [1, self.vtsfe.nb_frames, -1, self.vtsfe.vae_architecture["n_input"]])
                self.x_reconstr = x_reconstr
            # se shape = [nb_samples, nb_frames, n_input]
            
            se = self.get_reconstruction_squarred_error(x_samples, x_reconstr, transform_with_all_vtsfe=self.config.DATA_VISUALIZATION["transform_with_all_vtsfe"])
            # x_reconstr shape = [nb_sub_sequences, nb_samples, nb_frames, n_input]
            self.x_reconstr = np.transpose(self.x_reconstr, [0, 2, 1, 3])
            # reshaped se shape = [nb_mov_types, nb_samples_per_mov, nb_frames, n_input]
            #7*10*70*66 à partir de x_samples = 70*70*66
            self.se = np.reshape(se, [len(self.data_driver.mov_types), -1, self.vtsfe.nb_frames, self.vtsfe.vae_architecture["n_input"]])


    def destroy_vtsfe(self):
        if self.vtsfe is not None:
            self.vtsfe.session.close()
            tf.reset_default_graph()
            self.vtsfe = None


    def grid_search_dmp(self, data, save_path="default"):
        f = './grid_search/dmp.grid'
        for tau in np.arange(1., 2.25, 0.5):
            self.config.DMP_PARAMS["tau"] = tau
            for delta in np.arange(1., 2.25, 0.5):
                self.config.VTSFE_PARAMS["delta"] = delta
                for alpha in np.arange(0.5, 2.25, 0.5):
                    self.config.DMP_PARAMS["alpha"] = alpha
                    for beta in np.arange(0.5, 2.25, 0.5):
                        self.config.DMP_PARAMS["beta"] = beta
                        self.train(
                            data,
                            show_error=False,
                            resume=False,
                            save_path=save_path+"-dmp_grid-"+str(alpha)+"-"+str(beta)+"-"+str(tau)+"-"+str(delta)
                        )
                        self.data_driver.save_data(f, (alpha, beta, tau, delta), (self.global_error, alpha, beta, tau, delta, self.vtsfe.crashed))


    def grid_search_dmp_winners(self, path="default"):
        f = './grid_search/dmp.grid'
        winner = (1., 1., 0.5, 0.5)
        winning_ticket = 1000.
        first_value_winner = (1., 1., 0.5, 0.5)
        winning_first_value = 1000.
        for tau in np.arange(1., 2.25, 0.5):
            for delta in np.arange(1., 2.25, 0.5):
                for alpha in np.arange(0.5, 2.25, 0.5):
                    for beta in np.arange(0.5, 2.25, 0.5):
                        try:
                            global_error, alpha, beta, tau, delta, crashed = self.data_driver.read_data(f, (alpha, beta, tau, delta))
                            if crashed:
                                continue
                            if global_error[-1] < winning_ticket:
                                winner = (alpha, beta, tau, delta)
                                winning_ticket = global_error[-1]
                            if global_error[0] < winning_first_value:
                                first_value_winner = (alpha, beta, tau, delta)
                                winning_first_value = global_error[0]
                        except KeyError:
                            continue
        return (winner, winning_ticket, first_value_winner, winning_first_value)


    def show_reconstr_error_through_z_dims(self, dim_indices=range(2, 8), save_path="default"):
        mses = []
        mse_sigs = []
        errors = []
        default_dim_value = self.config.VTSFE_PARAMS["vae_architecture"]["n_z"]
        data = self.data_driver.get_whole_data_set()
        for dim in dim_indices:
            self.config.VTSFE_PARAMS["vae_architecture"]["n_z"] = dim
            self.train(
                data,
                show_error=False,
                resume=False,
                save_path=save_path+"-dim_"+str(dim)+".cpkt"
            )
            errors.append((
                self.global_error,
                self.vae_errors, self.vae_variances,
                self.vae_reconstr_errors, self.vae_reconstr_variances,
                self.vae_latent_errors, self.vae_latent_variances
            ))

            reconstr_error = self.get_reconstruction_squarred_error(transform_with_all_vtsfe=self.config.DATA_VISUALIZATION["transform_with_all_vtsfe"])
            mse = np.mean(reconstr_error)
            mse_sig = np.sqrt(np.var(reconstr_error))
            mses.append(mse)
            mse_sigs.append(mse_sig)

        fig = plt.figure(figsize=(50,100))
        fig.canvas.set_window_title("MSE on reconstruction with respect to latent space dimension")
        plt.errorbar(dim_indices, mses, yerr=mse_sigs, fmt='o')
        plt.show()

        for e in errors:
            self.plot_error(e)

        self.config.VTSFE_PARAMS["vae_architecture"]["n_z"] = default_dim_value


    def get_best_results_conditions(self, nb_blocks, nb_samples_per_mov, block_size, nb_train_blocks, nb_test_blocks, nb_eval_blocks, save_path="default-validation"):
        mses = []
        indices = []
        errors = []

        def extract_set(data, i, nb_blocks_in_set):
            index_i = i*block_size
            exceeding = index_i+nb_blocks_in_set - nb_samples_per_mov
            if exceeding < 0:
                exceeding = 0

            data_set = self.data_driver.to_set(
                [
                    self.data_driver.to_set(
                        data[:, :exceeding]
                    ),
                    self.data_driver.to_set(
                        data[:, index_i:index_i+nb_blocks_in_set]
                    ),
                ]
            )
            data_remains = np.concatenate(
                [
                    self.data_driver.to_set(data[:, exceeding:index_i]),
                    self.data_driver.to_set(data[:, index_i+nb_blocks_in_set:])
                ],
                axis=0
            )
            return data_set, data_remains

        data = self.data_driver.get_whole_data_set()
        for i in range(nb_blocks):
            training_set, training_remains = extract_set(data, i, nb_train_blocks)
            path = save_path+"-training_orig_"+str(i)
            self.train(
                # training_set shape = [nb_samples, nb_frames, n_input]]
                training_set,
                show_error=False,
                resume=False,
                save_path=path
            )

            if nb_test_blocks == 0:
                iterations = 1
            else:
                iterations = nb_blocks-nb_train_blocks
            for j in range(iterations):
                test_set, eval_set = extract_set(training_remains, j, nb_test_blocks)

                reconstr_error = self.get_reconstruction_squarred_error(eval_set, transform_with_all_vtsfe=self.config.DATA_VISUALIZATION["transform_with_all_vtsfe"])
                mse = np.mean(reconstr_error)
                mses.append(mse)
                indices.append((i, j))
                errors.append((self.global_error, self.vae_errors, self.vae_variances))


        mean_error = np.mean(mses)
        min_error = mses[0]
        index = (0, 0)
        for i, e in enumerate(mses):
            if e < min_error:
                min_error = e
                index = indices[i]
                error = errors[i]

        training_set, training_remains = extract_set(self.data_driver.data, index[0], nb_train_blocks)
        test_set, eval_set = extract_set(training_remains, index[1], nb_test_blocks)
        model_path = save_path+"-training_orig_"+str(index[0])

        return training_set, test_set, min_error, model_path, mean_error, error


    def leave_one_out(self, double_validation=False, resume=False):
        nb_samples_per_mov = self.data_driver.nb_samples_per_mov
        data = self.data_driver.data
        
        
        def extract_set(data, i, nb_blocks_in_set):
        #separe donnees data_set, data_remains
            index_i = i
            nb_samples = len(data[0])
            exceeding = index_i+nb_blocks_in_set - nb_samples
            if exceeding < 0:
                exceeding = 0

            data_set = np.concatenate(
                [
                    data[:, :exceeding],
                    data[:, index_i:index_i+nb_blocks_in_set]
                ],
                axis=1
            )
            data_remains = np.concatenate(
                [
                    data[:, exceeding:index_i],
                    data[:, index_i+nb_blocks_in_set:]
                ],
                axis=1
            )
            return data_set, data_remains

        winners = []
        
        #pour chaque mouvement test
        for test_index in range(nb_samples_per_mov):

            params = {
                "batch_size": 7,
                "nb_epochs": 60, #pourquoi pas 70 ?!
                "display_step": 1,
                "checkpoint_step": 10
            }

            if resume:
                resume_index = 6
                resume_index_started = False
                if resume_index_started and test_index == resume_index:
                    params.update({
                        "nb_epochs": 30
                    })

            if double_validation:
                test_data, train_eval_data = extract_set(data, test_index, 1)
                # if double cross-validation
                eval_mses = []
                for eval_index in range(nb_samples_per_mov):
                    if eval_index == test_index:
                        continue
                    rel_eval_index = eval_index
                    if eval_index > test_index:
                        rel_eval_index = eval_index-1
                    eval_data, training_data = extract_set(train_eval_data, rel_eval_index, 1)
                    flat_training_data = np.reshape(training_data, [-1, self.data_driver.nb_frames, self.data_driver.input_dim])
                    ticket = self.save_path+"_test_"+str(test_index)+"_eval_"+str(eval_index)
                    self.x_reconstr = None
                    self.config.TRAINING.update(params)
                    self.train(
                        # flat_training_data shape = [nb_samples, nb_frames, n_input]]
                        flat_training_data,
                        show_error=False,
                        resume=False,
                        save_path=ticket
                    )
                    
                    self.init_x_reconstr()
                    # se shape = [nb_mov_types, nb_samples_per_mov, nb_frames, n_input]
                    se = self.se[:, eval_index]
                    mse = np.mean(se)
                    eval_mses.append((mse, ticket, np.mean(self.se[:, test_index])))

                winner = winners[0][1]
                best_mse = winners[0][0]
                n = 0
                num = 0
                for mse,ticket,t in eval_mses:
                    if mse < best_mse:
                        best_mse = mse
                        winner = ticket
                        n = num
                    num += 1
                winners.append(eval_mses[n])
            else:
                # if simple cross-validation
                test_data, flat_training_data = self.data_driver.get_data_set([test_index], shuffle_samples=False)
                # flat_training_data = np.reshape(train_eval_data, [-1, self.data_driver.nb_frames, self.data_driver.input_dim])
                ticket = self.save_path+"_test_"+str(test_index)
                self.x_reconstr = None
                self.config.TRAINING.update(params)

                if resume:
                    if test_index < resume_index:
                        self.destroy_vtsfe()
                        restore_path = self.restore_path
                        self.restore_path = ticket
                        self.init_vtsfe()
                        self.restore_path = restore_path
                    if test_index == resume_index:
                        self.train(
                            # flat_training_data shape = [nb_samples, nb_frames, n_input]]
                            flat_training_data,
                            show_error=False,
                            resume=resume_index_started,
                            save_path=ticket
                        )
                    if test_index > resume_index:
                        self.train(
                            # flat_training_data shape = [nb_samples, nb_frames, n_input]]
                            flat_training_data,
                            show_error=False,
                            resume=False,
                            save_path=ticket
                        )
                else:
                    self.train(
                        # flat_training_data shape = [nb_samples, nb_frames, n_input]]
                        flat_training_data,
                        show_error=False,
                        resume=False,
                        save_path=ticket
                    )
                self.init_x_reconstr()
                # se shape = [nb_mov_types, nb_samples_per_mov, nb_frames, n_input]
                se = self.se[:, test_index]
                mse = np.mean(se)
                print("MSE = "+str(mse))
                winners.append((mse, ticket))

        if double_validation:
            mse = np.mean(np.array(np.array(winners)[:,2], dtype=np.float32))
        else:
            mse = np.mean(np.array(np.array(winners)[:,0], dtype=np.float32))

        # winner = winners[0][1]
        # best_mse = winners[0][0]
        # n = 0
        # for num,_,ticket,test_mse in enumerate(winners):
        #     if test_mse < best_mse:
        #         best_mse = test_mse
        #         winner = ticket
        #         n = num
        f = './leave-one-out/'+self.save_path
        self.data_driver.save_data(f, "mse", mse)
        self.data_driver.save_data(f, "winners", winners)

        return mse


    def get_leave_one_out_winner_full_path(self):
        f = './leave-one-out/'+self.save_path
        winner = None
        try:
            winners = self.data_driver.read_data(f, "winners")
            winner = winners[0][1]
            n = 0
            num = 0
            if len(winners[0]) > 2:
                # double cross-validation
                best_mse = winners[0][2]
                for _,ticket,test_mse in winners:
                    if test_mse < best_mse:
                        best_mse = test_mse
                        winner = ticket
                        n = num
                    num += 1
            else:
                # simple cross-validation
                best_mse = winners[0][0]
                for test_mse,ticket in winners:
                    if test_mse < best_mse:
                        best_mse = test_mse
                        winner = ticket
                        n = num
                    num += 1
        except KeyError:
            print("No leave-one-out made with that path.")
        return winner


    def show_reconstr_error_through_training_set_dim(self, train_proportions=None, save_path=None):
        nb_samples = self.data_driver.nb_samples_per_mov
        nb_blocks = nb_samples
        mean_mses = []
        min_mses = []
        errors = []

        if save_path is not None:
            path = save_path
        else:
            path = "validation"

        if train_proportions is None:
            iterations = range(1, nb_blocks-1)
        else:
            iterations = train_proportions

        for train_proportion in iterations:
            remains = nb_blocks - train_proportion
            block_size, nb_train_blocks, nb_test_blocks, nb_eval_blocks = self.data_driver.split(
                nb_blocks,
                nb_samples,
                train_proportion=train_proportion,
                test_proportion=0.,
                eval_proportion=remains
            )
            _, _, min_error, _, mean_error, error = self.get_best_results_conditions(
                nb_blocks,
                nb_samples,
                block_size,
                nb_train_blocks,
                nb_test_blocks,
                nb_eval_blocks,
                save_path=path
            )
            mean_mses.append(mean_error)
            min_mses.append(min_error)
            errors.append(error)

        fig = plt.figure(figsize=(50,100))
        fig.canvas.set_window_title("MSE on reconstruction with respect to training set dimension")
        plt.plot(range(1, nb_blocks-1), mean_mses)
        plt.scatter(range(1, nb_blocks-1), min_mses)
        plt.show()

        for e in errors:
            self.plot_error(e)





                        
    def retrieve_data_from_latent_space_ori(self,zs):
        self.init_vtsfe()
        x_reconstr_from_ls = self.vtsfe.reconstruct_fromLS(zs)
        return x_reconstr_from_ls

    def show_latent_space(self, sample_indices=None, data=None):
        self.init_vtsfe()
        if sample_indices is None:
            s_indices = range(self.data_driver.nb_samples_per_mov)
        else:
            s_indices = sample_indices
        if data is None:
            x_samples = self.data_driver.get_whole_data_set(shuffle_dataset=False)
        else:
            x_samples = data
        
        #retourne xs = 2 :
        zs = self.vtsfe.transform(x_samples, transform_with_all_vtsfe=self.config.DATA_VISUALIZATION["transform_with_all_vtsfe"])

        #print( "dans fonction " + str(len(zs)))
        self.vtsfe.show_latent_space(self.data_driver, zs,s_indices,"z",nb_samples_per_mov=10, displayed_movs=self.data_driver.mov_types, show_frames=False)

        if self.config.VTSFE_PARAMS["use_z_derivative"]:
            z_derivatives = self.vtsfe.transform_derivative(x_samples)
            self.vtsfe.show_latent_space(self.data_driver, z_derivatives, s_indices,"z derivative", nb_samples_per_mov=10, displayed_movs=self.data_driver.mov_types, show_frames=False  )
         
        return zs   
            

    def show_latent_space_recovered(self, zs, sample_indices=None, data=None):
        self.init_vtsfe()
        if sample_indices is None:
            s_indices = range(self.data_driver.nb_samples_per_mov)
        else:
            s_indices = sample_indices
        if data is None:
            x_samples = self.data_driver.get_whole_data_set(shuffle_dataset=False)
        else:
            x_samples = data
        
        #retourne xs = 2 :

        self.vtsfe.show_latent_space(self.data_driver, zs,s_indices,"z",nb_samples_per_mov=1, displayed_movs=self.data_driver.mov_types[0], show_frames=False)

     
        return zs   
            




    def show_reconstr_data(self, compare_to_other_models=True, sample_indices=None, only_hard_joints=True, average_reconstruction=True):
        self.init_vtsfe()
        if sample_indices is None:
            s_indices = range(self.data_driver.nb_samples_per_mov)
        else:
            s_indices = sample_indices

        x_samples = self.data_driver.get_whole_data_set(shuffle_dataset=False)
        # x_reconstr shape = [nb_sub_sequences, nb_samples, nb_frames, n_input]
        self.init_x_reconstr() #se = 7*10*70*66
        x_reconstr = [self.x_reconstr] #[(1, 70, 70, 66)] --> 10 testsx7actions, 70 frames, ...
        x_reconstr_names = [self.save_path] #['tighter_lb_light_joint_mvnx_2D_separated_encoder_variables_test_8']
        if compare_to_other_models:
            for lr in self.lrs:
                if lr.x_reconstr is not None:
                    x_reconstr.append(lr.x_reconstr)
                    x_reconstr_names.append(lr.save_path)
        x_reconstr = np.array(x_reconstr) #(1, 1, 70, 70, 66)
                                         #nb_tests_reconstr,         #nbTypeMvt xnbTestParMov, nbFrame, dimInput
                                         #1  #1   #7*10 # 70 #66
        x_reconstr = x_reconstr.reshape([len(x_reconstr_names), -1, len(self.data_driver.mov_types)*self.data_driver.nb_samples_per_mov, self.vtsfe.nb_frames, self.data_driver.input_dim])

        self.config.DATA_VISUALIZATION["reconstr_datasets"] = x_reconstr #(1, 1, 70, 70, 66)
        self.config.DATA_VISUALIZATION["reconstr_datasets_names"] = x_reconstr_names #['tighter_lb_light_joint_mvnx_2D_separated_encoder_variables_test_8']
        self.config.DATA_VISUALIZATION["x_samples"] = x_samples #(70,70,66)
        self.config.DATA_VISUALIZATION["sample_indices"] = s_indices #8
        self.config.DATA_VISUALIZATION["only_hard_joints"] = only_hard_joints
        self.vtsfe.show_data(
            self.config.DATA_VISUALIZATION
        )
        return x_reconstr
       
    #def show_reconstr_data_from_zs(self,zs, compare_to_other_models=True, sample_indices=None, only_hard_joints=True, average_reconstruction=True):
        #self.init_vtsfe()
        #if sample_indices is None:
            #s_indices = range(self.data_driver.nb_samples_per_mov)
        #else:
            #s_indices = sample_indices

        #x_samples = self.data_driver.get_whole_data_set(shuffle_dataset=False)f

        ## x_reconstr shape = [nb_sub_sequences, nb_samples, nb_frames, n_input]
        #self.init_x_reconstr_zs(zs) #se_zs ? = 7*10*70*66
        #x_reconstr = [self.x_reconstr]
        #x_reconstr_names = [self.save_path]

        #x_reconstr = np.array(x_reconstr)
        #x_reconstr = x_reconstr.reshape([len(x_reconstr_names), -1, len(self.data_driver.mov_types)*self.data_driver.nb_samples_per_mov, self.vtsfe.nb_frames, self.data_driver.input_dim])

        ##self.config.DATA_VISUALIZATION["reconstr_datasets"] = x_reconstr
        ##self.config.DATA_VISUALIZATION["reconstr_datasets_names"] = x_reconstr_names
        ##self.config.DATA_VISUALIZATION["x_samples"] = x_samples
        ##self.config.DATA_VISUALIZATION["sample_indices"] = s_indices
        ##self.config.DATA_VISUALIZATION["only_hard_joints"] = only_hard_joints
        ##self.vtsfe.show_data(
            ##self.config.DATA_VISUALIZATION
        ##)
        
        
    def show_reconstr_data_ori(self,x_mean, x_log_sigma_sqs, type_indices=None, sample_indices=None, only_hard_joints=True, average_reconstruction=True):


        self.init_vtsfe()

        if sample_indices is None:
            s_indices = range(self.data_driver.nb_samples_per_mov)
        else:
            s_indices = sample_indices
        x_samples = self.data_driver.get_whole_data_set(shuffle_dataset=False)
       
        # x_reconstr shape = [nb_sub_sequences, nb_samples, nb_frames, n_input]
        self.init_x_reconstr() #se = 7*10*70*66
        x_reconstr = [self.x_reconstr] #[(1, 70, 70, 66)] --> 10 testsx7actions, 70 frames, ...
        x_reconstr_names = [self.save_path] #['tighter_lb_light_joint_mvnx_2D_separated_encoder_variables_test_8']
        x_reconstr = np.array(x_reconstr) #(1, 1, 70, 70, 66)
                                         #nb_tests_reconstr,         #nbTypeMvt xnbTestParMov, nbFrame, dimInput
                                         #1  #1   #7*10 # 70 #66
        x_reconstr = x_reconstr.reshape([len(x_reconstr_names), -1, len(self.data_driver.mov_types)*self.data_driver.nb_samples_per_mov, self.vtsfe.nb_frames, self.data_driver.input_dim])

        self.config.DATA_VISUALIZATION["reconstr_datasets"] = x_reconstr #(1, 1, 70, 70, 66)
        self.config.DATA_VISUALIZATION["reconstr_datasets_names"] = x_reconstr_names #['tighter_lb_light_joint_mvnx_2D_separated_encoder_variables_test_8']
        self.config.DATA_VISUALIZATION["x_samples"] = x_samples #(70,70,66)
        self.config.DATA_VISUALIZATION["sample_indices"] = s_indices #8
        self.config.DATA_VISUALIZATION["displayed_movs"] = type_indices #0
        self.config.DATA_VISUALIZATION["only_hard_joints"] = only_hard_joints

        self.vtsfe.show_data_ori(
            x_samples,
            self.config.DATA_VISUALIZATION
        )
 

    def show_inferred_parameters(self, data=None):
        self.init_vtsfe()
        if data is None:
            x_samples = self.data_driver.get_whole_data_set(shuffle_dataset=False)
        else:
            x_samples = data
        self.vtsfe.model.show_inferred_parameters(
            self.data_driver,
            x_samples,
            nb_samples_per_mov=1,
            displayed_movs=self.data_driver.mov_types
        )

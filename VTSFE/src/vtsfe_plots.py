import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pdb

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import numpy as np

markers = [
    'o', # 'circle',
    'D', # 'diamond',
    '*', # 'star',
    'd', # 'thin_diamond',
    's', # 'square',
    'X', # 'x_filled',
    'p', # 'pentagon',
    '8', # 'octagon',
    'x', # 'x',
    'h', # 'hexagon1',
    'H', # 'hexagon2',
    '+', # 'plus',
    '|', # 'vline',
    '_', # 'hline',
    '<', # 'triangle_left',
    '>', # 'triangle_right',
    '1', # 'tri_down',
    '2', # 'tri_up',
    '3', # 'tri_left',
    '4', # 'tri_right',
    'P', # 'plus_filled',
    0, # 'tickleft',
    1, # 'tickright',
    2, # 'tickup',
    3, # 'tickdown',
    4, # 'caretleft',
    5, # 'caretright',
    6, # 'caretup',
    7, # 'caretdown',
    8, # 'caretleftbase',
    9, # 'caretrightbase',
    10, # 'caretupbase',
    11, # 'caretdownbase'
]

font_size = 12
ylim_low = -0.25
ylim_up = 1


class VTSFE_Plots():
    """plotting methods for VTSFE"""

    def __init__(self, vtsfe):
        self.vtsfe = vtsfe


    def save_fig(self, plt, title):
        if self.vtsfe.save_path is None:
            s_path = "default"
        else:
            s_path = self.vtsfe.save_path
        plt.savefig("./img/"+s_path+"-"+title+".svg")


    def record_animation(self, anim, title):
        if self.vtsfe.save_path is None:
            s_path = "default"
        else:
            s_path = self.vtsfe.save_path
       #anim.save("./animations/"+s_path+"-"+title+".mp4", fps=15, extra_args=['-vcodec', 'libx264'])


    def plot_mse(self, se_dataset, se_names, displayed_movs, s_indices):
        # vse for dynamics evaluation
        # vses = np.var(se, axis=2)
        # TODO: mettre dans une fonction de plot séparée ou comme option d'affichage en fin de celle-ci puisque contrairement à la MSE, on ne s'intéresse pas à la VSE à travers le temps, mais au travers des dimensions => sommer toutes les vses
        # mse on inputs through time

        def get_sigma_bounds(errors, variances):
            sigmas = np.sqrt(variances)
            sigma_sup = np.add(errors, sigmas)
            sigma_inf = np.subtract(errors, sigmas)
            return sigma_inf, sigma_sup
        nb_ses = len(se_names) #type de tests
        colors = cm.rainbow(np.linspace(0, 1, nb_ses+1))
        mses_per_mvt_and_frame = []
        vses_per_mvt_and_frame = []
        mses_per_frame = []
        vses_per_frame = []
        fig, ax = plt.subplots(figsize=(20,10))
        fig.canvas.set_window_title("MSE")
        ax.set_title("MSE on all movements")
        ax.set_xlabel('Time')
        ax.set_ylabel('Squarred error')
        ax.grid()
        for j in range(nb_ses): #pour chaque test
            se = se_dataset[j, range(len(displayed_movs)), s_indices]
            frame_dim = se.shape[-2:]

            if len(s_indices) == 1:
                se = np.reshape(se, [len(displayed_movs), 1, frame_dim[0], frame_dim[1]])

            mse_per_mvt_and_frame_and_sample = np.mean(se, axis=3)
            vse_per_mvt_and_frame_and_sample = np.var(se, axis=3)

            mses_per_mvt_and_frame.append(
                np.mean(mse_per_mvt_and_frame_and_sample, axis=1)
            )
            vses_per_mvt_and_frame.append(
                np.add(
                    np.mean(vse_per_mvt_and_frame_and_sample, axis=1),
                    np.var(mse_per_mvt_and_frame_and_sample, axis=1)
                )
            )
            mses_per_frame.append(
                np.mean(mses_per_mvt_and_frame[j], axis=0)
            )
            vses_per_frame.append(
                np.add(
                    np.mean(vses_per_mvt_and_frame[j], axis=0),
                    np.var(mses_per_mvt_and_frame[j], axis=0)
                )
            )
            print("\n-------- "+se_names[j])
            print("Total MSE = "+str(np.mean(mses_per_frame[j])))
            print("VSE sum = "+str(np.sum(np.var(se, axis=2))))

            ax.plot(
                range(self.vtsfe.nb_frames),
                mses_per_frame[j],
                c=colors[j],
                linestyle='solid',
                label=se_names[j]
            )
            sigma_bounds = get_sigma_bounds(mses_per_frame[j], vses_per_frame[j])
            ax.fill_between(range(self.vtsfe.nb_frames), sigma_bounds[0], sigma_bounds[1], alpha=0.5, label="Std dev", facecolor=colors[j])
        ax.legend(bbox_to_anchor=(1.1, 1.1), loc=1, borderaxespad=0.)
        plt.ion()
        plt.show()
        plt.pause(.001)        
        mses_per_mvt_and_frame = np.array(mses_per_mvt_and_frame)
        mses_per_frame = np.array(mses_per_frame)
        print(mses_per_mvt_and_frame.shape)
        print(mses_per_frame.shape)

        for i, mov in enumerate(displayed_movs):
            fig, ax = plt.subplots(figsize=(20,10))
            fig.canvas.set_window_title("MSE of "+mov)
            ax.set_title("MSE of "+mov)
            ax.set_xlabel('Time')
            ax.set_ylabel('Squarred error')
            ax.grid()

            for j in range(nb_ses):
                # ax.plot(
                #     range(self.vtsfe.nb_frames),
                #     mses_per_frame[j],
                #     c=colors[j],
                #     linestyle='dashed',
                #     label=se_names[j]
                # )
                ax.plot(
                    range(self.vtsfe.nb_frames),
                    mses_per_mvt_and_frame[j][i],
                    c=colors[j],
                    linestyle='solid',
                    label=se_names[j]
                )
                sigma_bounds = get_sigma_bounds(mses_per_mvt_and_frame[j][i], vses_per_mvt_and_frame[j][i])
                ax.fill_between(range(self.vtsfe.nb_frames), sigma_bounds[0], sigma_bounds[1], alpha=0.5, label="Std dev", facecolor=colors[j])

            ax.legend(bbox_to_anchor=(1.1, 1.1), loc=1, borderaxespad=0.)
            plt.ion()
            plt.show()
            plt.pause(.001)


    def plot_error(self,
        global_error,
        vae_errors, vae_variances,
        vae_reconstr_errors, vae_reconstr_variances,
        vae_model_errors, vae_model_variances,
        vae_latent_errors, vae_latent_variances
    ):
        nb_vaes = len(vae_errors[0])
        nb_epochs = len(global_error)

        variables = []
        labels = [
            "Global mean error (all VAEs)",
            "Mean error /VAE",
            "Mean reconstr error /VAE",
            "Mean model error /VAE",
            "Mean latent error /VAE",
            "Std dev /VAE",
            "Std reconstr dev /VAE",
            "Std model dev /VAE",
            "Std latent dev /VAE"
        ]
        nb_error_types = 4
        linestyles = ["dashed"] + ["-"]*nb_error_types
        colors = cm.rainbow(np.linspace(0, 1, nb_error_types+1))
        vaes_labels = (
            ["Origin VAE"]
            + ["Standard VAE frame = "+str(i) for i in range(len(self.vtsfe.std_vaes_like_origin))]
            + ["Model VAE frame = "+str(i) for i in range(len(self.vtsfe.std_vaes_like_origin), len(self.vtsfe.std_vaes_like_origin) + self.vtsfe.sub_sequences_size-len(self.vtsfe.std_vae_indices))]
            + ["Goal VAE"]
        )

        def set_information(raw_variables):
            for v, var in enumerate(raw_variables):
                variables.append({
                    "values": np.array(var),
                    "label": labels[v],
                    "linestyle": linestyles[v],
                    "color": colors[v]
                })

        def get_sigma_bounds(errors, variances):
            sigmas = np.sqrt(variances)
            sigma_sup = np.add(errors, sigmas)
            sigma_inf = np.subtract(errors, sigmas)
            return sigma_inf, sigma_sup

        set_information([
            global_error,
            vae_errors,
            vae_reconstr_errors,
            vae_model_errors,
            vae_latent_errors
        ])

        sigma_bounds = get_sigma_bounds(vae_errors, vae_variances)
        sigma_bounds_reconstr = get_sigma_bounds(vae_reconstr_errors, vae_reconstr_variances)
        sigma_bounds_model = get_sigma_bounds(vae_model_errors, vae_model_variances)
        sigma_bounds_latent = get_sigma_bounds(vae_latent_errors, vae_latent_variances)

        column_size = int(np.sqrt(nb_vaes))
        plots_mod = nb_vaes % column_size
        row_size = int(nb_vaes / column_size)
        if plots_mod != 0:
            row_size += 1

        fig, axes = plt.subplots(column_size, row_size, sharex=True, sharey=True, figsize=(20,10))
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
            if label is not None:
                ax.legend(loc=(-0.7, 0.65))

        for i in range(nb_vaes):
            ax = axes[i]
            ax.grid()
            ax.margins(0.)
            ax.set_title(vaes_labels[i])
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Error')

            if i == 0:
                label = labels[-1]
            else:
                label = None
            # plot a colored surface between mu-sigma and mu+sigma
            ax.fill_between(range(nb_epochs), sigma_bounds[0][:, i], sigma_bounds[1][:, i], alpha=0.5, label=labels[1], facecolor=colors[1])
            ax.fill_between(range(nb_epochs), sigma_bounds_reconstr[0][:, i], sigma_bounds_reconstr[1][:, i], alpha=0.5, label=labels[2], facecolor=colors[2])
            ax.fill_between(range(nb_epochs), sigma_bounds_model[0][:, i], sigma_bounds_model[1][:, i], alpha=0.5, label=labels[3], facecolor=colors[3])
            ax.fill_between(range(nb_epochs), sigma_bounds_latent[0][:, i], sigma_bounds_latent[1][:, i], alpha=0.5, label=labels[4], facecolor=colors[4])
            # plot global and local means
            for k, var in enumerate(variables):
                if k != 0:
                    var["indices"] = (slice(0, None), i)
                var.update({
                    "ax": ax
                })
                plot_variable(**var)
                var["label"] = None
        self.save_fig(plt, "training_errors")
        plt.ion()
        plt.show()
        plt.pause(.001)


    def show_latent_space(self, data_driver, latent_data, sample_indices, title, zs_inf=[], displayed_movs=[], nb_samples_per_mov=1, show_frames=True, titleFrame=None):
        """
            Latent space representation
        """
        labels = data_driver.data_labels #10*7 (7 =nbType)

        # don't display more movements than there are
        if nb_samples_per_mov > len(sample_indices):
            display_per_mov = len(sample_indices)
        else:
            display_per_mov = nb_samples_per_mov

        # translate observations in latent space
        # latent_data shape = [nb_sub_sequences, nb_frames, nb_samples, n_z]
        z_mu = np.array(latent_data)

        print("input space dimensions = "+str(self.vtsfe.vae_architecture["n_input"]))
        print("latent space dimensions = "+str(self.vtsfe.vae_architecture["n_z"]))

        # indices_per_mov_type = []
        # for mov in displayed_movs:
        #     mov_index = data_driver.mov_indices[mov]
        #     indices_per_mov_type.append(np.where(labels == mov_index))

        nb_mov_types = len(displayed_movs)

        colors = cm.rainbow(np.linspace(0, 1, nb_mov_types+1))

        # if either latent space isn't in 3-D or you want to see the time axis
        if self.vtsfe.vae_architecture["n_z"] != 3 or show_frames:
            if show_frames:
                # if you want to see evolution through time,
                # plot the necessary number of 3-D plots with an axis for time, and 2 for latent space

                def plot_sample(ax, s, i, flat_index, x, y, marker=None, markevery=None, label=None):
                    ax.plot(
                            range(self.vtsfe.nb_frames),
                            z_mu[s, :, flat_index, x],
                            z_mu[s, :, flat_index, y],
                            c=colors[i],
                            label=label,
                            marker=marker,
                            markevery=markevery
                        )
                    if label is not None:
                        ax.legend(bbox_to_anchor=(-0.1, 1.5), loc=1, borderaxespad=0.)

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
                            self.vtsfe.nb_frames-1,
                            z_mu[s, self.vtsfe.nb_frames-1, flat_index, x],
                            z_mu[s, self.vtsfe.nb_frames-1, flat_index, y],
                            c=colors[i],
                            marker='v',
                            s=320
                        )

                # turn on interactive mode
                plt.ion()

                # plot every permutation of 2 dimensions of latent space
                for x in range(self.vtsfe.vae_architecture["n_z"]):
                    for y in range(x+1, self.vtsfe.vae_architecture["n_z"]):
                        fig = plt.figure(figsize=(20,10))
                        #fig, axes = plt.subplots(column_size, row_size, sharex=True, sharey=True, figsize=(20,10))
                        fig.canvas.set_window_title("Latent space - "+title)
                        #fig.set_title("Latent space dim="+str(self.vtsfe.vae_architecture["n_z"])+" - "+title)
                        ax = fig.gca(projection='3d')
                        ax.set_title("Latent space", fontsize=font_size)
                        ax.tick_params(labelsize=font_size)
                        ax.set_xlabel("$z_"+str(x)+"$", fontsize=font_size)
                        ax.set_ylabel("$z_"+str(y)+"$", fontsize=font_size)

                        # plot a path through time per sample, a color per movement type
                        for i, mov in enumerate(displayed_movs):
                            for j in range(display_per_mov):
                                flat_index = data_driver.mov_indices[mov] + sample_indices[j]
                                for s in range(len(z_mu)):
                                    # add only one instance of the same label
                                    if x == 0 and y == 1 and j == 0 and s == 0:
                                        plot_sample(ax, s, i, flat_index, x, y, marker='o', markevery=slice(1, self.vtsfe.nb_frames-1, 1), label=labels[flat_index])
                                    else:
                                        plot_sample(ax, s, i, flat_index, x, y, marker='o', markevery=slice(1, self.vtsfe.nb_frames-1, 1))

                                    add_markers(ax, s, i, flat_index, x, y)
                        
                        if(len(zs_inf)>0):
                            ax.plot(range(self.vtsfe.nb_frames),zs_inf[:,0], zs_inf[:,1], c=colors[nb_mov_types+1], label='Infered latent space', marker='o', markevery=slice(1, self.vtsfe.nb_frames-1, 1))
                    if label is not None:
                        ax.legend(bbox_to_anchor=(-0.1, 1.5), loc=1, borderaxespad=0.)

                        plt.legend()
                        plt.ion()
                        plt.show()
                        plt.pause(.001)
                        _ = input("Press [enter] to continue.") # wait for input from the user
                        plt.close()    # close the figure to show the next one.

                # turn off interactive mode
                plt.ioff()

            else:
                # if you don't want to see evolution through time,
                # plot the necessary number of 2-D plots with 2 axes for latent space

                def plot_sample(ax, s, i, flat_index, x, y, marker=None, markevery=None, label=None):
                    ax.plot(
                            z_mu[s, :, flat_index, x], #z_mu = 1*70*70*2 - zmu[0|1 , 70 timestep, numeroTest (=indxtype+numEchantillon), x ]
                            z_mu[s, :, flat_index, y],
                            c=colors[i],
                            label=label,
                            marker=markers[i],
                            markevery=markevery,
                            linewidth=3.0,
                            markersize=12
                        )
                    if label is not None:
                        ax.legend(bbox_to_anchor=(-0.1, 1.5), loc=1, borderaxespad=0.)

                def add_markers(ax, s, i, flat_index, x, y):
                    # mark start of sequence
                    ax.scatter(
                            z_mu[s, 0, flat_index, x],
                            z_mu[s, 0, flat_index, y],
                            c=colors[i],
                            marker='^',
                            s=400
                        )
                    # mark end of sequence
                    ax.scatter(
                            z_mu[s, self.vtsfe.nb_frames-1, flat_index, x],
                            z_mu[s, self.vtsfe.nb_frames-1, flat_index, y],
                            c=colors[i],
                            marker='v',
                            s=400
                        )

                # plot every permutation of 2 dimensions of latent space in the same figure
                # wrap these configurations in a 1-D array named 'plots' to use with the 1-D array of axes
                plots = []
                for x in range(self.vtsfe.vae_architecture["n_z"]):
                    for y in range(x+1, self.vtsfe.vae_architecture["n_z"]):
                        plots.append({
                            "x": x,
                            "y": y
                        })

                column_size = int(np.sqrt(len(plots)))
                plots_mod = len(plots) % column_size
                row_size = int(len(plots) / column_size)
                if plots_mod != 0:
                    row_size += 1

                fig, axes = plt.subplots(column_size, row_size, sharex=True, sharey=True, figsize=(20,10))
                fig.canvas.set_window_title("Latent space - "+title)

                if len(plots) == 1:
                    axes = np.array([axes])
                else:
                    axes = axes.reshape([-1])

                for p, plot in enumerate(plots):
                    ax = axes[p]
                    ax.set_title("Latent space", fontsize=font_size)
                    ax.tick_params(labelsize=font_size)
                    ax.set_xlabel("$z_"+str(plot["x"])+"$", fontsize=font_size)
                    ax.set_ylabel("$z_"+str(plot["y"])+"$", fontsize=font_size)
                    # plot a path per sample, a color per movement type
                    for i, mov in enumerate(displayed_movs): # i :0-7 ; mov = bent; bent_strongly;..
                        for j in range(display_per_mov): #nb mouvement representé (tous=10 ou sous ensemble si precisé)
                            flat_index = data_driver.mov_indices[mov] + sample_indices[j] # base_echantillon_a_repr + num echantillon
                            for s in range(len(z_mu)): # s : 1-2 (x,y)
                                # add only one instance of the same label
                                if p == 0 and j == 0 and s == 0:
                                    plot_sample(ax, s, i, flat_index, plot["x"], plot["y"], markevery=slice(1, self.vtsfe.nb_frames-1, 1), label=labels[flat_index])
                                else:
                                    plot_sample(ax, s, i, flat_index, plot["x"], plot["y"], markevery=slice(1, self.vtsfe.nb_frames-1, 1))

                                add_markers(ax, s, i, flat_index, plot["x"], plot["y"])
                    if(len(zs_inf)>0):
                        ax.plot(zs_inf[:,0], zs_inf[:,1], c=colors[nb_mov_types], label='Infered latent space', marker='+', markevery=slice(1, self.vtsfe.nb_frames-1, 1),linewidth=1.0, markersize=12)
                        ax.scatter(zs_inf[0, 0], zs_inf[0, 1], c=colors[nb_mov_types], marker='^', s=400)
                        ax.scatter(zs_inf[self.vtsfe.nb_frames-1, 0], zs_inf[self.vtsfe.nb_frames-1, 1], c=colors[nb_mov_types], marker='v', s=400)

                plt.legend(prop={'size':font_size})

                if (titleFrame is None):
                    titleFrame = "latent_space_"+ str(sample_indices)

                self.save_fig(plt,titleFrame)

                plt.ion()
                plt.show()

                plt.pause(.001)
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
                if label is not None:
                    ax.legend(bbox_to_anchor=(-0.1, 1.5), loc=1, borderaxespad=0.)

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
                        z_mu[s, self.vtsfe.nb_frames-1, flat_index, 0],
                        z_mu[s, self.vtsfe.nb_frames-1, flat_index, 1],
                        z_mu[s, self.vtsfe.nb_frames-1, flat_index, 2],
                        c=colors[i],
                        marker='v',
                        s=320
                    )

            fig = plt.figure(figsize=(20,10))
            fig.canvas.set_window_title("Latent space - "+title)
            ax = fig.gca(projection='3d')
            # plot a path per sample, a color per movement type
            for i, mov in enumerate(displayed_movs):
                for j in range(display_per_mov):
                    flat_index = data_driver.mov_indices[mov] + sample_indices[j]
                    for s in range(len(z_mu)):
                        # add only one instance of the same label
                        if j == 0 and s == 0:
                            plot_sample(ax, s, i, flat_index, marker='o', markevery=slice(1, self.vtsfe.nb_frames-1, 1), label=labels[flat_index])
                        else:
                            plot_sample(ax, s, i, flat_index, marker='o', markevery=slice(1, self.vtsfe.nb_frames-1, 1))

                        add_markers(ax, s, i, flat_index)

            plt.legend()
            plt.ion()
            plt.show()
            plt.pause(.001)

    def show_data(self, data_driver, reconstr_datasets, reconstr_datasets_names, sample_indices, x_samples, plot_variance=False, nb_samples_per_mov=1, show=True, displayed_movs=None, dynamic_plot=False, plot_3D=False, window_size=10, time_step=5, body_lines=True, only_hard_joints=True, transform_with_all_vtsfe=True, average_reconstruction=True, data_inf=[]):
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
        print(str(display_per_mov))
        if transform_with_all_vtsfe:
            nb_sub_sequences = self.vtsfe.nb_sub_sequences
        else:
            nb_sub_sequences = 1

        nb_colors = display_per_mov +1
        if len(reconstr_datasets) > 0:
            nb_colors *= len(reconstr_datasets)
        colors = cm.rainbow(np.linspace(0, 1, nb_colors))
        for j,reco in enumerate(reconstr_datasets):
            # reco shape = [nb_sub_sequences, nb_samples, nb_frames, n_input]
            print("\n-------- "+reconstr_datasets_names[j])
            print("Dynamics (Sum of variances through time) = "+str(np.sum(np.var(reco, axis=2))))

        if plot_3D:
            # to avoid modifying x_samples
            data = np.copy(x_samples) #70.70.66
            # x_samples shape = [nb_samples, nb_frames, segment_count, 3 (coordinates)]
            data = data.reshape([len(data_driver.mov_types)*data_driver.nb_samples_per_mov, self.vtsfe.nb_frames, -1, 3]) #70.70.22.3
            
            #TODO: Add variables to defin the shape of data_inf

            if(data_inf != []): 
                tmp_shape = data_inf.shape
                data_inf = data_inf.reshape(tmp_shape[0],tmp_shape[1],23,3)

                if(tmp_shape[0]== 7):
                    data_inf_tmp = np.zeros([70,70,23,3])
                    for i in np.arange(7):
                        data_inf_tmp[i*10+8] = data_inf[i]
                    data_inf = data_inf_tmp
                    
            data_reconstr = []
            for j,reco in enumerate(reconstr_datasets): #1.1.70.70.66
                data_reconstr.append(reco.reshape([nb_sub_sequences, len(data_driver.mov_types)*data_driver.nb_samples_per_mov, self.vtsfe.nb_frames, -1, 3]))
            segment_count = len(data[0, 0])#22

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
                        if(data_inf != []):
                            plot_body_lines(plot_inf[index : next_index], sample_indices[j], k, data_inf)

                        for r,reco in enumerate(data_reconstr):
                            for sub in range(nb_sub_sequences):
                                plot_body_lines(plot_recs[r*nb_sub_sequences + sub][index : next_index], r*nb_sub_sequences + sample_indices[j], k, reco[sub])
                        index = next_index
                    title.set_text("Time = {}".format(k))
                    if not show:
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
                    if not show:
                        ax.view_init(30, -150 + 0.7 * k)

            if displayed_movs is not None:
                # turn on interactive mode
                plt.ion()
                # scatter all movements in displayed_movs, a color per movement sample
                for i, mov in enumerate(displayed_movs):
                    plots = []# plot data original
                    plot_recs = [] # plot data reconstr
                    plot_inf = [] # plot data inf

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
                        #Prepare bodylines for the x_original plot
                        for j in range(display_per_mov):
                            # plot the nb_body_lines+nb_additional_lines lines of body segments
                            for k in range(nb_body_lines + nb_additional_lines):
                                # plots shape = [display_per_mov*(nb_body_lines + nb_additional_lines)]
                                plots.append(ax.plot([], [], [], c=colors[j], marker='o')[0])
   
                        #Prepare bodylines for the x inf plot
                        for j in range(display_per_mov):
                        # plot the nb_body_lines+nb_additional_lines lines of body segments
                            for k in range(nb_body_lines + nb_additional_lines):
                                # plots shape = [display_per_mov*(nb_body_lines + nb_additional_lines)]
                                plot_inf.append(ax.plot([], [], [], c=colors[display_per_mov], marker='D', linestyle='dashed')[0])
    

                        # Prepare bodylines for the x reconstr plot
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

                    plt.legend()
                    # call the animator.  blit=True means only re-draw the parts that have changed.
                    anim = animation.FuncAnimation(fig, animate, frames=self.vtsfe.nb_frames, interval=time_step, blit=False)
                    # Save as mp4. This requires mplayer or ffmpeg to be installed
                    self.record_animation(anim, mov+"-input_data_space")
                    if show:
                        plt.show()
                        _ = input("Press [enter] to continue.") # wait for input from the user
                    plt.close()    # close the figure to show the next one.

                # turn off interactive mode
                plt.ioff()
        else:
            # if it's not a 3D plot

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
                if linestyle == "-":
                    m = None
                else:
                    m = markers[j]
                plots.append(
                    ax.plot(
                        [],
                        [],
                        c=colors[j],
                        linestyle=linestyle,
                        label=label,
                        linewidth=3.0,
                        marker=m,
                        markersize=12,
                        markevery=4
                    )[0]
                )

                if label is not None:
                    ax.legend(bbox_to_anchor=(2, 1.6), loc=1, borderaxespad=0.)
                return plots

            plots = []
            jump = nb_sub_sequences*len(reconstr_datasets) + 1
            def plot_source(t, s, source):
                mov_count = s*display_per_mov*jump
                for j in range(display_per_mov):
                    flat_index = data_driver.mov_indices[mov] + sample_indices[j]
                    flat_index_plots = mov_count + jump*j

                    # plot input data
                    plots[flat_index_plots].set_data(
                        range(t, t + ws ),
                        x_samples[flat_index, t: t + ws, source]
                    )
                    flat_index_plots += 1
                    for r,reco in enumerate(reconstr_datasets):
                        flat_r_index = r*nb_sub_sequences
                        for sub in range(nb_sub_sequences):
                            # plot reconstructed data
                            plots[flat_index_plots + flat_r_index + sub].set_data(
                                range(t, t + ws ),
                                reco[sub, flat_index, t: t + ws, source]
                            )

            if len(data_driver.data_types) == 1:
                dim_names_available = True
            else:
                dim_names_available = False

            if dynamic_plot:
                # turn on interactive mode
                plt.ion()
                ws = window_size
            else:
                ws = self.vtsfe.nb_frames

            for i, mov in enumerate(displayed_movs):
                if dynamic_plot:
                    fig, axes = plt.subplots(column_size, row_size, sharex=True, sharey=True, figsize=(20,10))
                    fig.canvas.set_window_title("Input data space of "+mov)
                    if sources_count == 1:
                        axes = np.array([axes])
                    else:
                        axes = axes.reshape([-1])
                plots = []
                # for each source type
                # wrap all plots in a 1-D array named 'plots' for the animation function
                for s,source in enumerate(source_list):
                    if dynamic_plot:
                        ax = axes[s]
                    else:
                        fig, ax = plt.subplots(figsize=(20,10))
                        fig.canvas.set_window_title("Input data space of "+mov)
                        plots = []
                        s = 0
                    ax.set_xlim(0, ws-1)
                    ax.tick_params(labelsize=font_size)
                    ax.set_ylim(ylim_low, ylim_up)
                    ax.set_xlabel('Time', fontsize=font_size)
                    dim = data_driver.dim_names[source][-1]
                    if dim not in ['x', 'y', 'z']:
                        dim = "data dimension"
                    ax.set_ylabel("Normalized "+dim, fontsize=font_size)
                    ax.grid()
                    if dim_names_available:
                        ax.set_title(data_driver.dim_names[source], fontsize=font_size)
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
                                        lab = reconstr_datasets_names[r]
                                        if 'tighter' in reconstr_datasets_names[r]:
                                            lab = 'VTSFE light (II)'
                                        elif 'vae_dmp' in reconstr_datasets_names[r]:
                                            lab = 'VAE-DMP (III)'
                                        elif 'vae_only' in reconstr_datasets_names[r]:
                                            lab = 'VAE only (IV)'
                                        # reconstructed data plot
                                        plots = add_plot(plots, ax, flat_r_index, label=lab, linestyle="dashed")
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
                    if not dynamic_plot:
                        plot_source(0, s, source)
                        plt.legend(fontsize=font_size)
                        self.save_fig(plt, mov+"-"+data_driver.dim_names[source].replace('/', '-'))
                        if show:
                            plt.ion()
                            plt.show()
                            plt.pause(.001)

                # animation function.  This is called sequentially
                def animate(t):
                    for s,source in enumerate(source_list):
                        ax = axes[s]
                        ax.set_xlim(t, t + ws - 1)
                        plot_source(t, s, source)
                    return plots

                if dynamic_plot:
                    plt.legend()
                    # call the animator.  blit=True means only re-draw the parts that have changed.
                    anim = animation.FuncAnimation(fig, animate, frames=self.vtsfe.nb_frames-window_size, interval=time_step, blit=False)
                    # Save as mp4. This requires mplayer or ffmpeg to be installed
                    self.record_animation(anim, mov+"-input_data_space")
                    if show:
                        plt.show()
                        _ = input("Press [enter] to continue.") # wait for input from the user
                    plt.close()    # close the figure to show the next one.

            if dynamic_plot:
                # turn off interactive mode
                plt.ioff()


    def show_data_ori2(self, data_driver, reconstr_datasets, reconstr_datasets_names, sample_indices, x_samples, plot_variance=False, nb_samples_per_mov=1, show=True, displayed_movs=None, dynamic_plot=False, plot_3D=False, window_size=10, time_step=5, body_lines=True, only_hard_joints=True, transform_with_all_vtsfe=True, average_reconstruction=True, data_inf=[]):
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
            nb_sub_sequences = self.vtsfe.nb_sub_sequences
        else:
            nb_sub_sequences = 1

        nb_colors = display_per_mov
        if len(reconstr_datasets) > 0:
            nb_colors *= len(reconstr_datasets)
        colors = cm.rainbow(np.linspace(0, 1, nb_colors))
        for j, reco in enumerate(reconstr_datasets):
            # reco shape = [nb_sub_sequences, nb_samples, nb_frames, n_input]
            print("\n-------- "+reconstr_datasets_names[j])
            print("Dynamics (Sum of variances through time) = "+str(np.sum(np.var(reco, axis=2))))

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
            if linestyle == "-":
                m = None
            else:
                m = markers[j]
            plots.append(
                ax.plot(
                    [],
                    [],
                    c=colors[j],
                    linestyle=linestyle,
                    label=label,
                    linewidth=3.0,
                    marker=m,
                    markersize=12,
                    markevery=4
                )[0]
            )

            if label is not None:
                ax.legend(bbox_to_anchor=(2, 1.6), loc=1, borderaxespad=0.)
            return plots

        plots = []
        jump = nb_sub_sequences*len(reconstr_datasets) + 1
        def plot_source(t, s, source):
            mov_count = s*display_per_mov*jump
            for j in range(display_per_mov):
                flat_index = data_driver.mov_indices[mov] + sample_indices[j]
                flat_index_plots = mov_count + jump*j

                # plot input data
                plots[flat_index_plots].set_data(
                    range(t, t + ws ),
                    x_samples[flat_index, t: t + ws, source]
                )
                flat_index_plots += 1
                for r,reco in enumerate(reconstr_datasets):
                    flat_r_index = r*nb_sub_sequences
                    for sub in range(nb_sub_sequences):
                        # plot reconstructed data
                        plots[flat_index_plots + flat_r_index + sub].set_data(
                            range(t, t + ws ),
                            reco[sub, flat_index, t: t + ws, source]
                        )

        if len(data_driver.data_types) == 1:
            dim_names_available = True
        else:
            dim_names_available = False

        if dynamic_plot:
            # turn on interactive mode
            plt.ion()
            ws = window_size
        else:
            ws = self.vtsfe.nb_frames

        #for i, mov in enumerate(displayed_movs):
        i = 0, 
        mov = displayed_movs[0] # ajout ori : j'affiche que le premier exemple pour que ce ne soitpas lou
        if dynamic_plot:
            fig, axes = plt.subplots(column_size, row_size, sharex=True, sharey=True, figsize=(20,10))
            fig.canvas.set_window_title("Input data space of "+mov)
            if sources_count == 1:
                axes = np.array([axes])
            else:
                axes = axes.reshape([-1])
        plots = []
        # for each source type
        # wrap all plots in a 1-D array named 'plots' for the animation function
        s = 9
        source = source_list[9]
        #for s,source in enumerate(source_list):
        if dynamic_plot:
            ax = axes[s]
        else:
            fig, ax = plt.subplots(figsize=(20,10))
            fig.canvas.set_window_title("Input data space of "+mov)
            plots = []
            s = 0
        ax.set_xlim(0, ws-1)
        ax.tick_params(labelsize=font_size)
        ax.set_ylim(ylim_low, ylim_up)
        ax.set_xlabel('Time', fontsize=font_size)
        dim = data_driver.dim_names[source][-1]
        if dim not in ['x', 'y', 'z']:
            dim = "data dimension"
        ax.set_ylabel("Normalized "+dim, fontsize=font_size)
        ax.grid()
        if dim_names_available:
            ax.set_title(data_driver.dim_names[source], fontsize=font_size)
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
                            lab = reconstr_datasets_names[r]
                            if 'tighter' in reconstr_datasets_names[r]:
                                lab = 'VTSFE light (II)'
                            elif 'vae_dmp' in reconstr_datasets_names[r]:
                                lab = 'VAE-DMP (III)'
                            elif 'vae_only' in reconstr_datasets_names[r]:
                                lab = 'VAE only (IV)'
                            # reconstructed data plot
                            plots = add_plot(plots, ax, flat_r_index, label=lab, linestyle="dashed")
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
        if not dynamic_plot:
            plot_source(0, s, source)
            plt.legend(fontsize=font_size)
            self.save_fig(plt, mov+"-"+data_driver.dim_names[source].replace('/', '-'))
            if show:
                plt.ion()
                plt.show()
                plt.pause(.001)
        #fin for
        # animation function.  This is called sequentially
        def animate(t):
            for s,source in enumerate(source_list):
                ax = axes[s]
                ax.set_xlim(t, t + ws - 1)
                plot_source(t, s, source)
            return plots

        if dynamic_plot:
            plt.legend()
            # call the animator.  blit=True means only re-draw the parts that have changed.
            anim = animation.FuncAnimation(fig, animate, frames=self.vtsfe.nb_frames-window_size, interval=time_step, blit=False)
            # Save as mp4. This requires mplayer or ffmpeg to be installed
            self.record_animation(anim, mov+"-input_data_space")
            if show:
                plt.ion()
                plt.show()
                plt.pause(.001)
                _ = input("Press [enter] to continue.") # wait for input from the user
            plt.close()    # close the figure to show the next one.

        if dynamic_plot:
            # turn off interactive mode
            plt.ioff()


    def show_data_ori(self, x_mean, data_driver, reconstr_datasets, reconstr_datasets_names, sample_indices, x_samples, plot_variance=False, nb_samples_per_mov=1, show=True, displayed_movs='bent_fw', dynamic_plot=False, plot_3D=False, window_size=10, time_step=5, body_lines=True, only_hard_joints=True, transform_with_all_vtsfe=True, average_reconstruction=True, data_inf=[]):
        """
            Data space representation.
            You can optionally plot reconstructed data as well at the same time.
            - displayed_movs : type de mouvement à représenter (ex: ['bent_fw','window_open'])
        """
        labels = data_driver.data_labels
        # don't display more movements than there are
        if nb_samples_per_mov > len(sample_indices):
            display_per_mov = len(sample_indices)
        else:
            display_per_mov = nb_samples_per_mov

        if transform_with_all_vtsfe:
            nb_sub_sequences = self.vtsfe.nb_sub_sequences
        else:
            nb_sub_sequences = 1

        nb_colors = display_per_mov + 1
        if len(reconstr_datasets) > 0:
            nb_colors *= len(reconstr_datasets)
        colors = cm.rainbow(np.linspace(0, 1, nb_colors))
        for j,reco in enumerate(reconstr_datasets):
            # reco shape = [nb_sub_sequences, nb_samples, nb_frames, n_input]
            print("\n-------- "+reconstr_datasets_names[j])
            print("Dynamics (Sum of variances through time) = "+str(np.sum(np.var(reco, axis=2))))


        #on affiche seulement les joints ci dessous
        source_list = [4,28,40,52,64]#joints l3-l5 ; rigthForeArm- ; leftForeArm ; RigthFoot ; LeftFoot
        sources_count = len(source_list)
        

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
          #  pdb.set_trace()
            if linestyle == "-":
                m = None
            else:
                m = markers[j]
            plots.append(
                ax.plot(
                    [],
                    [],
                    c=colors[j],
                    linestyle=linestyle,
                    label=label,
                    linewidth=3.0,
                    marker=m,
                    markersize=12,
                    markevery=4
                )[0]
            )

            if label is not None:
                ax.legend(bbox_to_anchor=(2, 1.6), loc=1, borderaxespad=0.)
            return plots

        plots = []
        jump = nb_sub_sequences*len(reconstr_datasets) + 1
        def plot_source(t, s, source):

            mov_count = 0# s*display_per_mov*jump TODO essayer de comprendre ce que c'est
            for j in range(display_per_mov):
                flat_index = data_driver.mov_indices[mov] + sample_indices[j]
                flat_index_plots = mov_count + jump*j
                
                # plot input data
                plots[flat_index_plots].set_data(
                    range(t, t + ws ),
                    x_samples[flat_index, t: t + ws, source]
                )
                flat_index_plots += 1

                #plot predicted data
                plots[flat_index_plots].set_data(
                    range(t, t + ws ),
                    x_mean[t: t + ws, source]
                )

                flat_index_plots += 1

                for r,reco in enumerate(reconstr_datasets):
                    flat_r_index = r*nb_sub_sequences
                    for sub in range(nb_sub_sequences):
                        # plot reconstructed data
                        plots[flat_index_plots + flat_r_index + sub].set_data(
                            range(t, t + ws ),
                            reco[sub, flat_index, t: t + ws, source]
                        )

        if len(data_driver.data_types) == 1:
            dim_names_available = True
        else:
            dim_names_available = False

        ws = self.vtsfe.nb_frames


        for i, mov in enumerate(displayed_movs):
     
            plots = []
            # for each source type
            # wrap all plots in a 1-D array named 'plots' for the animation function
            fig, ax = plt.subplots(5,figsize=(20,10))
            ax[0].set_title("Input data space of "+mov)
            
            for s in range(len(source_list)):
                source = source_list[s]
     
                
                plots = []

                ax[s].set_xlim(0, ws-1)
                ax[s].tick_params(labelsize=font_size)
                ax[s].set_ylim(ylim_low, ylim_up)
                ax[s].set_xlabel('Time', fontsize=font_size)
                dim = data_driver.dim_names[source][-1]
                if dim not in ['x', 'y', 'z']:
                    dim = "data dimension"

                if dim_names_available:
                    ax[s].set_ylabel("Normalized "+data_driver.dim_names[source], fontsize=font_size)
                ax[s].grid()
                # plot all movements types in same plot, each with a different color
                # plot display_per_mov samples of the same movement type with the same color
                for j in range(display_per_mov):
                    flat_index = data_driver.mov_indices[mov] + sample_indices[j]

                    if j == 0 and s == 0:
                        # input data plot
                        plots = add_plot(plots, ax[s], j, label=labels[flat_index])
                        # inference data plot
                        plots = add_plot(plots, ax[s], nb_colors-1, linestyle=":", label='inferenceFromProMPs')
                        for r,reco in enumerate(reconstr_datasets):
                            for sub in range(nb_sub_sequences):
                                flat_r_index = r*nb_sub_sequences + j
                                if sub == 0:
                                    
                                    lab = reconstr_datasets_names[r]
                                    if 'tighter' in reconstr_datasets_names[r]:
                                        lab = 'VTSFE light (II)'
                                    elif 'vae_dmp' in reconstr_datasets_names[r]:
                                        lab = 'VAE-DMP (III)'
                                    elif 'vae_only' in reconstr_datasets_names[r]:
                                        lab = 'VAE only (IV)'
                                    # reconstructed data plot
                                    plots = add_plot(plots, ax[s], flat_r_index, label=lab, linestyle="dashed")
                                else:
                                    # reconstructed data plot
                                    plots = add_plot(plots, ax[s], flat_r_index, linestyle="dashed")
                    else:
                        
                        # input data plot
                        plots = add_plot(plots, ax[s], j)
                        # inference data plot
                        plots = add_plot(plots, ax[s], nb_colors-1, linestyle=":")
                        for r,reco in enumerate(reconstr_datasets):
                            for sub in range(nb_sub_sequences):
                                # reconstructed data plot
                                plots = add_plot(plots, ax[s], r*nb_sub_sequences + j, linestyle="dashed")
                if not dynamic_plot:
                    plot_source(0, s, source)
                    plt.legend(fontsize=font_size)
                    self.save_fig(plt, mov+"-"+data_driver.dim_names[source].replace('/', '-'))
                    if show:
                        plt.ion()
                        plt.show()
                        plt.pause(.001)
            ## animation function.  This is called sequentially
            #def animate(t):
                #for s,source in enumerate(source_list):
                    #ax = axes[s]
                    #ax.set_xlim(t, t + ws - 1)
                    #plot_source(t, s, source)
                #return plots

            #if dynamic_plot:
                #plt.legend()
                ## call the animator.  blit=True means only re-draw the parts that have changed.
                #anim = animation.FuncAnimation(fig, animate, frames=self.vtsfe.nb_frames-window_size, interval=time_step, blit=False)
                ## Save as mp4. This requires mplayer or ffmpeg to be installed
                #self.record_animation(anim, mov+"-input_data_space")
                #if show:
                    #plt.ion()
                    #plt.show()
                    #plt.pause(.001)
                    #_ = input("Press [enter] to continue.") # wait for input from the user
                #plt.close()    # close the figure to show the next one.

        #if dynamic_plot:
            ## turn off interactive mode
            #plt.ioff()

        ###ax.set_title("MSE on all movements")
        ###ax.set_xlabel('Time')
        ###ax.set_ylabel('Squarred error')
        
        
        ### to avoid modifying x_samples
        ##data = np.copy(x_samples)
        ###data = np.zeros((70, 70,5))
        ##if(len(data)== 70):
            ##for tyype in range(1):      
                ##fig2, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(5,1)
                ##fig2.canvas.set_window_title("data_"+displayed_movs[tyype] )      
                ##for aaction in sample_indices:#range(10):
                  ###  for tmp2 in range(70):
                    ###pdb.set_trace()
                    ##ax1.plot(data[(tyype*10) + aaction][1:70][0])
                    ##ax2.plot(data[(tyype*10) + aaction][1:70][1])
                    ##ax3.plot(data[(tyype*10) + aaction][1:70][2])
                    ##ax4.plot(data[(tyype*10) + aaction][1:70][3])
                    ##ax5.plot(data[(tyype*10) + aaction][1:70][4])                                               
         ###       data[tmp][tmp2] = np.copy(x_samples[tmp][tmp2][0:5])
     
        #labels = data_driver.data_labels
        #pdb.set_trace()
        ## don't display more movements than there are
        #if nb_samples_per_mov > len(sample_indices):
            #display_per_mov = len(sample_indices)
        #else:
            #display_per_mov = nb_samples_per_mov

        #if transform_with_all_vtsfe:
            #nb_sub_sequences = self.vtsfe.nb_sub_sequences
        #else:
            #nb_sub_sequences = 1

        #nb_colors = display_per_mov
        #if len(reconstr_datasets) > 0:
            #nb_colors *= len(reconstr_datasets)
        #colors = cm.rainbow(np.linspace(0, 1, nb_colors))
        #for j,reco in enumerate(reconstr_datasets):
            ## reco shape = [nb_sub_sequences, nb_samples, nb_frames, n_input]
            #print("\n-------- "+reconstr_datasets_names[j])
            #print("Dynamics (Sum of variances through time) = "+str(np.sum(np.var(reco, axis=2))))

        #if only_hard_joints:
            #source_list = data_driver.hard_dimensions
            #sources_count = len(source_list)
        #else:
            #sources_count = len(x_samples[0, 0])
            #source_list = range(sources_count)

        #column_size = int(np.sqrt(sources_count))
        #plots_mod = sources_count % column_size
        #row_size = int(sources_count / column_size)
        #if plots_mod != 0:
            #row_size += 1

        ## add a plot to animate
        #def add_plot(plots, ax, j, linestyle="-", label=None):
            ## plots shape = [sources_count*displayed_movs*display_per_mov*(1+nb_sub_sequences)]
            ## or
            ## plots shape = [sources_count*displayed_movs*display_per_mov]
            #if linestyle == "-":
                #m = None
            #else:
                #m = markers[j]
            #plots.append(
                #ax.plot(
                    #[],
                    #[],
                    #c=colors[j],
                    #linestyle=linestyle,
                    #label=label,
                    #linewidth=3.0,
                    #marker=m,
                    #markersize=12,
                    #markevery=4
                #)[0]
            #)

            #if label is not None:
                #ax.legend(bbox_to_anchor=(2, 1.6), loc=1, borderaxespad=0.)
            #return plots

        #plots = []
        #jump = nb_sub_sequences*len(reconstr_datasets) + 1
        #def plot_source(t, s, source):
            #mov_count = s*display_per_mov*jump
            #for j in range(display_per_mov):
                #flat_index = data_driver.mov_indices[mov] + sample_indices[j]
                #flat_index_plots = mov_count + jump*j

                ## plot input data
                #plots[flat_index_plots].set_data(
                    #range(t, t + ws ),
                    #x_samples[flat_index, t: t + ws, source]
                #)
                #flat_index_plots += 1
                #for r,reco in enumerate(reconstr_datasets):
                    #flat_r_index = r*nb_sub_sequences
                    #for sub in range(nb_sub_sequences):
                        ## plot reconstructed data
                        #plots[flat_index_plots + flat_r_index + sub].set_data(
                            #range(t, t + ws ),
                            #reco[sub, flat_index, t: t + ws, source]
                        #)

        #if len(data_driver.data_types) == 1:
            #dim_names_available = True
        #else:
            #dim_names_available = False

        #if dynamic_plot:
            ## turn on interactive mode
            #plt.ion()
            #ws = window_size
        #else:
            #ws = self.vtsfe.nb_frames

        ##for i, mov in enumerate(displayed_movs):
        #i = 0, 
        #mov = displayed_movs[0]
        #if dynamic_plot:
            #fig, axes = plt.subplots(column_size, row_size, sharex=True, sharey=True, figsize=(20,10))
            #fig.canvas.set_window_title("Input data space of "+mov)
            #if sources_count == 1:
                #axes = np.array([axes])
            #else:
                #axes = axes.reshape([-1])
        #plots = []
        ## for each source type
        ## wrap all plots in a 1-D array named 'plots' for the animation function
        #s = 9
        #source = source_list[9]
        ##for s,source in enumerate(source_list):
        #if dynamic_plot:
            #ax = axes[s]
        #else:
            #fig, ax = plt.subplots(figsize=(20,10))
            #fig.canvas.set_window_title("Input data space of "+mov)
            #plots = []
            #s = 0
        #ax.set_xlim(0, ws-1)
        #ax.tick_params(labelsize=font_size)
        #ax.set_ylim(ylim_low, ylim_up)
        #ax.set_xlabel('Time', fontsize=font_size)
        #dim = data_driver.dim_names[source][-1]
        #if dim not in ['x', 'y', 'z']:
            #dim = "data dimension"
        #ax.set_ylabel("Normalized "+dim, fontsize=font_size)
        #ax.grid()
        #if dim_names_available:
            #ax.set_title(data_driver.dim_names[source], fontsize=font_size)
        ## plot all movements types in same plot, each with a different color
        ## plot display_per_mov samples of the same movement type with the same color
        #for j in range(display_per_mov):
            #flat_index = data_driver.mov_indices[mov] + sample_indices[j]

            #if j == 0 and s == 0:
                ## input data plot
                #plots = add_plot(plots, ax, j, label=labels[flat_index])

                #for r,reco in enumerate(reconstr_datasets):
                    #for sub in range(nb_sub_sequences):
                        #flat_r_index = r*nb_sub_sequences + j
                        #if sub == 0:
                            #lab = reconstr_datasets_names[r]
                            #if 'tighter' in reconstr_datasets_names[r]:
                                #lab = 'VTSFE light (II)'
                            #elif 'vae_dmp' in reconstr_datasets_names[r]:
                                #lab = 'VAE-DMP (III)'
                            #elif 'vae_only' in reconstr_datasets_names[r]:
                                #lab = 'VAE only (IV)'
                            ## reconstructed data plot
                            #plots = add_plot(plots, ax, flat_r_index, label=lab, linestyle="dashed")
                        #else:
                            ## reconstructed data plot
                            #plots = add_plot(plots, ax, flat_r_index, linestyle="dashed")
            #else:
                ## input data plot
                #plots = add_plot(plots, ax, j)

                #for r,reco in enumerate(reconstr_datasets):
                    #for sub in range(nb_sub_sequences):
                        ## reconstructed data plot
                        #plots = add_plot(plots, ax, r*nb_sub_sequences + j, linestyle="dashed")
        #if not dynamic_plot:
            #plot_source(0, s, source)
            #plt.legend(fontsize=font_size)
            #self.save_fig(plt, mov+"-"+data_driver.dim_names[source].replace('/', '-'))
            #if show:
                #plt.ion()
                #plt.show()
                #plt.pause(.001)
        ##fin for
        ## animation function.  This is called sequentially
        #def animate(t):
            #for s,source in enumerate(source_list):
                #ax = axes[s]
                #ax.set_xlim(t, t + ws - 1)
                #plot_source(t, s, source)
            #return plots

        #if dynamic_plot:
            #plt.legend()
            ## call the animator.  blit=True means only re-draw the parts that have changed.
            #anim = animation.FuncAnimation(fig, animate, frames=self.vtsfe.nb_frames-window_size, interval=time_step, blit=False)
            ## Save as mp4. This requires mplayer or ffmpeg to be installed
            #self.record_animation(anim, mov+"-input_data_space")
            #if show:
                #plt.ion()
                #plt.show()
                #plt.pause(.001)
                #_ = input("Press [enter] to continue.") # wait for input from the user
            #plt.close()    # close the figure to show the next one.

        #if dynamic_plot:
            ## turn off interactive mode
            #plt.ioff()

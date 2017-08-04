# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

from src.launcher import Launcher

from app.vae_dmp_joint_mvnx_2D import vae_dmp_joint_mvnx_2D
from app.vae_dmp_joint_mvnx_5D import vae_dmp_joint_mvnx_5D

from app.tighter_lb_joint_mvnx_2D_L15 import tighter_lb_joint_mvnx_2D_L15
from app.tighter_lb_joint_mvnx_5D_L15 import tighter_lb_joint_mvnx_5D_L15

from app.tighter_lb_joint_mvnx_2D_L30 import tighter_lb_joint_mvnx_2D_L30
from app.tighter_lb_joint_mvnx_5D_L30 import tighter_lb_joint_mvnx_5D_L30

from app.tighter_lb_light_joint_mvnx_2D import tighter_lb_light_joint_mvnx_2D
from app.tighter_lb_light_joint_mvnx_5D import tighter_lb_light_joint_mvnx_5D


trainings = []

#######################################################################################################################################################
# EXPERIMENTS   #######################################################################################################################################
# trainings.append(("dmp-dvbf-2D-nutan_nutanDB", vae_dmp_joint_chendb_2D)) # Pending
# trainings.append(("dmp-dvbf-5D-nutan_nutanDB", vae_dmp_joint_chendb_5D)) # Pending

# trainings.append(("dmp-dvbf-params_nutan_model_joint_mvnx_2D", vae_dmp_joint_mvnx_2D))  # OK
# trainings.append(("dmp-dvbf-params_tighter_lower_bound_without_attractor_joint_mvnx_2D_light", tighter_lb_light_joint_mvnx_2D))   # OK
# trainings.append(("dmp-dvbf-params_tighter_lower_bound_without_attractor_joint_mvnx_2D", tighter_lb_joint_mvnx_2D_L15))   # OK
# trainings.append(("tighter_lb_joint_mvnx_2D_L30", tighter_lb_joint_mvnx_2D_L30))   # Pending

# trainings.append(("dmp-dvbf-params_nutan_model_joint_mvnx_5D", vae_dmp_joint_mvnx_5D))  # OK
# trainings.append(("dmp-dvbf-params_tighter_lower_bound_without_attractor_joint_mvnx_5D_light", tighter_lb_light_joint_mvnx_5D))   # OK
# trainings.append(("dmp-dvbf-params_tighter_lower_bound_without_attractor_joint_mvnx_5D", tighter_lb_joint_mvnx_5D_L15))   # OK
trainings.append(("tighter_lb_joint_mvnx_5D_L30", tighter_lb_joint_mvnx_5D_L30))   # Pending

#######################################################################################################################################################
# USELESS    ##########################################################################################################################################
# trainings.append(("dmp-dvbf-params_tighter_lower_bound_without_attractor_joint_mvnx_2D_light_with_xreconstr_sigma", params_tighter_lower_bound_without_attractor_joint_mvnx_2D_light_with_xreconstr_sigma))   # OK
# trainings.append(("dmp-dvbf-2D-without_z_derivative_nutanDB", params_without_z_derivative_nutanDB_2D))    # Pending
#######################################################################################################################################################
# GRID SEARCH   #######################################################################################################################################
# trainings.append(("dmp-dvbf-no_sequence_encoders-optimize_all-without_z_derivative-2D_grid_search", params))
#######################################################################################################################################################
# DEPRECATED    #######################################################################################################################################
# trainings.append(("dmp-dvbf-2D-nutan", params_nutan_model_mvnx_data_2D))  # OK
# trainings.append(("dmp-dvbf-5D-nutan", params_nutan_model_mvnx_data_5D))  # OK
# trainings.append(("dmp-dvbf-params_without_attractor_mvnx_2D", params_without_attractor_mvnx_2D))   # OK
# trainings.append(("dmp-dvbf-params_without_attractor_mvnx_5D", params_without_attractor_mvnx_5D))   # OK
# trainings.append(("dmp-dvbf-params_tighter_lower_bound_without_attractor_mvnx_2D", params_tighter_lower_bound_without_attractor_mvnx_2D))   # OK
# trainings.append(("dmp-dvbf-params_tighter_lower_bound_without_attractor_mvnx_5D", params_tighter_lower_bound_without_attractor_mvnx_5D))   # OK
#######################################################################################################################################################


restore = False
train = True
lrs = []
for i, training in enumerate(trainings):
    restore_path = None
    if restore:
        restore_path = training[0]

    lr = Launcher(
        training[1],
        lrs,
        restore_path=restore_path,
        save_path=training[0]
    )

    if restore and not train:
        lr.config.DMP_PARAMS["model_monte_carlo_sampling"] = 1
        lr.config.DMP_PARAMS["prior_monte_carlo_sampling"] = 1
        lr.config.VTSFE_PARAMS["vae_architecture"]["L"] = 1

    # winners = lr.grid_search_dmp_winners()
    # print(winners)

    lr.config.TRAINING.update({
        "batch_size": 6,
        "nb_epochs": 60,
        "display_step": 1,
        "checkpoint_step": 10
    })
    # if i == 0:
    #     lr.config.TRAINING["nb_epochs"] = 9
    # if i == 1:
    #     lr.config.TRAINING["nb_epochs"] = 50

    nb_training_samples = 6
    data, remains = lr.data_driver.get_data_set(nb_training_samples, shuffle_samples=False)
    # lr.show_data(
    #     sample_indices=slice(nb_training_samples+1, None),
    #     only_hard_joints=True
    # )

    if train:
        lr.train(
            data,
            show_error=False,
            resume=restore,
            save_path=training[0]
        )
    else:
        lr.config.DATA_VISUALIZATION.update({
            "nb_samples_per_mov": 1,
            "window_size": 15,
            "time_step": 5,
            "transform_with_all_vtsfe": False,
            "plot_3D": False,
            "body_lines": True
        })

        lr.plot_error(training[0])
        # lr.show_latent_space(
            # sample_indices=range(nb_training_samples, lr.data_driver.nb_samples_per_mov)
        # )

        lr.config.DATA_VISUALIZATION["displayed_movs"] = ["kicking", "window_open", "lifting_box"]
        if i == 2:
            lr.show_reconstr_data(
                compare_to_other_models=True,
                sample_indices=range(nb_training_samples, lr.data_driver.nb_samples_per_mov),
                only_hard_joints=True
            )
        else:
            lr.init_x_reconstr()

        # print("------------- Test set :")
        # lr.plot_variance_histogram(remains, 4)

        # lr.show_inferred_parameters()

        # lr.grid_search_dmp(data)

        # lr.show_reconstr_error_through_z_dims(
        #     dim_indices=range(2, 8),
        #     save_path=restore_path
        # )

        # lr.show_reconstr_error_through_training_set_dim(
        # )
    lrs.append(lr)
    lr.destroy_vtsfe()

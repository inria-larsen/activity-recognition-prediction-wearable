# -*- coding: utf-8 -*-
import numpy as np
import pdb
import os
#import yarp
import sys
import os
from launcher import Launcher
#from src.launcher import Launcher

# config files
## joints
#MVNX c'est le format des fichier Xsens MVN
from app.vae_only_joint_mvnx_2D import vae_only_joint_mvnx_2D
from app.vae_only_joint_mvnx_5D import vae_only_joint_mvnx_5D
from app.vae_only_joint_mvnx_7D import vae_only_joint_mvnx_7D

from app.vae_dmp_no_z_derivative_joint_mvnx_2D_separated import vae_dmp_no_z_derivative_joint_mvnx_2D_separated
from app.vae_dmp_no_z_derivative_joint_mvnx_2D import vae_dmp_no_z_derivative_joint_mvnx_2D

from app.vae_dmp_joint_mvnx_2D_separated import vae_dmp_joint_mvnx_2D_separated
from app.vae_dmp_joint_mvnx_5D_separated import vae_dmp_joint_mvnx_5D_separated
from app.vae_dmp_joint_mvnx_7D_separated import vae_dmp_joint_mvnx_7D_separated

# from app.vae_dmp_joint_mvnx_5D import vae_dmp_joint_mvnx_5D
# from app.vae_dmp_joint_mvnx_7D import vae_dmp_joint_mvnx_7D

# from app.tighter_lb_joint_mvnx_2D_L15 import tighter_lb_joint_mvnx_2D_L15
# from app.tighter_lb_joint_mvnx_5D_L15 import tighter_lb_joint_mvnx_5D_L15
# from app.tighter_lb_joint_mvnx_7D_L15 import tighter_lb_joint_mvnx_7D_L15

# from app.tighter_lb_joint_mvnx_2D_L30 import tighter_lb_joint_mvnx_2D_L30
# from app.tighter_lb_joint_mvnx_5D_L30 import tighter_lb_joint_mvnx_5D_L30
# from app.tighter_lb_joint_mvnx_7D_L30 import tighter_lb_joint_mvnx_7D_L30

from app.vae_dmp_joint_mvnx_2D import vae_dmp_joint_mvnx_2D
from app.tighter_lb_light_joint_mvnx_2D import tighter_lb_light_joint_mvnx_2D
from app.tighter_lb_joint_mvnx_2D_L15_separated import tighter_lb_joint_mvnx_2D_L15_separated
from app.tighter_lb_joint_mvnx_5D_L15_separated import tighter_lb_joint_mvnx_5D_L15_separated
from app.tighter_lb_joint_mvnx_2D_L30_separated import tighter_lb_joint_mvnx_2D_L30_separated

from app.tighter_lb_light_joint_mvnx_2D_separated import tighter_lb_light_joint_mvnx_2D_separated
from app.tighter_lb_light_joint_mvnx_5D_separated import tighter_lb_light_joint_mvnx_5D_separated
from app.tighter_lb_light_joint_mvnx_7D_separated import tighter_lb_light_joint_mvnx_7D_separated

# from app.tighter_lb_light_joint_mvnx_5D import tighter_lb_light_joint_mvnx_5D
# from app.tighter_lb_light_joint_mvnx_7D import tighter_lb_light_joint_mvnx_7D

## positions
# from app.vae_dmp_position_mvnx_2D import vae_dmp_position_mvnx_2D
# from app.vae_dmp_position_mvnx_5D import vae_dmp_position_mvnx_5D
# from app.vae_dmp_joint_mvnx_7D import vae_dmp_joint_mvnx_7D

# from app.tighter_lb_position_mvnx_2D_L15 import tighter_lb_position_mvnx_2D_L15
# from app.tighter_lb_position_mvnx_5D_L15 import tighter_lb_position_mvnx_5D_L15
# from app.tighter_lb_joint_mvnx_7D_L15 import tighter_lb_joint_mvnx_7D_L15

# from app.tighter_lb_joint_mvnx_2D_L30 import tighter_lb_joint_mvnx_2D_L30
# from app.tighter_lb_joint_mvnx_5D_L30 import tighter_lb_joint_mvnx_5D_L30
# from app.tighter_lb_joint_mvnx_7D_L30 import tighter_lb_joint_mvnx_7D_L30

# from app.tighter_lb_light_position_mvnx_2D import tighter_lb_light_position_mvnx_2D
# from app.tighter_lb_light_position_mvnx_5D import tighter_lb_light_position_mvnx_5D
# from app.tighter_lb_light_joint_mvnx_7D import tighter_lb_light_joint_mvnx_7D


trainings = []

#######################################################################################################################################################
# EXPERIMENTS   #######################################################################################################################################

# Whether you want to run a leave_one_out in training mode or not
### TO DO WHAT? ###
leave_one_out = False
# Select the sample index in [0, 9] on which models will be tested (all other samples will be used for training)
# Given our 7 movement types, this means: 9*7 = 63 training samples, 1*7 = 7 test samples
test_index = 8
if leave_one_out:
    ti = ''
else:
    ti = "_test_"+str(test_index)

#trainings.append(("vae_only_joint_mvnx_2D"+ti, vae_only_joint_mvnx_2D))    # OK
# trainings.append(("vae_only_joint_mvnx_5D"+ti, vae_only_joint_mvnx_5D))    # OK
# trainings.append(("vae_only_joint_mvnx_7D"+ti, vae_only_joint_mvnx_7D))    # OK

# trainings.append(("vae_dmp_no_z_derivative_joint_mvnx_2D_separated_encoder_variables", vae_dmp_no_z_derivative_joint_mvnx_2D_separated))  # OK
# trainings.append(("vae_dmp_no_z_derivative_joint_mvnx_2D", vae_dmp_no_z_derivative_joint_mvnx_2D))  # OK

#trainings.append(("vae_dmp_joint_mvnx_2D_separated_encoder_variables"+ti, vae_dmp_joint_mvnx_2D_separated))  # OK
# trainings.append(("vae_dmp_joint_mvnx_5D_separated_encoder_variables"+ti, vae_dmp_joint_mvnx_5D_separated))  # OK
# trainings.append(("vae_dmp_joint_mvnx_7D_separated_encoder_variables"+ti, vae_dmp_joint_mvnx_7D_separated))  # OK

trainings.append(("tighter_lb_light_joint_mvnx_2D_separated_encoder_variables"+ti, tighter_lb_light_joint_mvnx_2D_separated))   # OK
# trainings.append(("tighter_lb_light_joint_mvnx_5D_separated_encoder_variables"+ti, tighter_lb_light_joint_mvnx_5D_separated))   # OK
# trainings.append(("tighter_lb_light_joint_mvnx_7D_separated_encoder_variables"+ti, tighter_lb_light_joint_mvnx_7D_separated))   # OK

# trainings.append(("tighter_lb_light_joint_mvnx_2D_all_test_8", tighter_lb_light_joint_mvnx_2D))   # OK
# trainings.append(("vae_dmp_joint_mvnx_2D_all", vae_dmp_joint_mvnx_2D))   # Pending
# trainings.append(("vae_dmp_joint_mvnx_2D_all_test_8", vae_dmp_joint_mvnx_2D))   # OK
# trainings.append(("tighter_lb_joint_mvnx_2D_L15_separated_encoder_variables_test_8", tighter_lb_joint_mvnx_2D_L15_separated))   # OK
# trainings.append(("tighter_lb_joint_mvnx_2D_L30_separated_encoder_variables_test_8", tighter_lb_joint_mvnx_2D_L30_separated))   # OK
# trainings.append(("tighter_lb_joint_mvnx_2D_L15_P15_separated_encoder_variables_test_8", tighter_lb_joint_mvnx_2D_L15_separated))   # Killed


###################################
### POurquoi tu ne visualise qu'un mouvement ? C'est pas censé etre une distribution ?###
DATA_VISUALIZATION = {
    "nb_samples_per_mov": 1,
    "window_size": 15,
    "time_step": 5,
    "transform_with_all_vtsfe": False,
    "average_reconstruction": False,
    "plot_3D": False,
    "body_lines": True,
    "dynamic_plot": False,
    "show": True
}


# show input data space
show_data = False
# plot learning errors through epochs
plot_error = False
plot_mse = False
# show latent space
show_latent_space = True
# plot the global MSE and MSE for each movement type
# show reconstruction of input data space, compare it between models in trainings
show_reconstr_data = False
# movement types shown at data reconstruction
reconstr_data_displayed_movs = ["kicking"]

restore = True
train = False
lrs = []
mses = []
for i, training in enumerate(trainings):
    ### c'est quoi restore path? ###
    restore_path = None
    if restore:
        restore_path = training[0]
                #(self, config, lrs, restore_path=None, save_path=None)
    lr = Launcher(training[1], lrs, restore_path=restore_path, save_path=training[0])

    if restore and not train:
        lr.config.DMP_PARAMS["model_monte_carlo_sampling"] = 1
        lr.config.DMP_PARAMS["prior_monte_carlo_sampling"] = 1
        lr.config.VTSFE_PARAMS["vae_architecture"]["L"] = 1

    lr.config.TRAINING.update({
        "batch_size": 7,
        "nb_epochs": 10,
        "display_step": 1,
        "checkpoint_step": 10
    })

    training_indices = list(range(test_index))+list(range(test_index+1,10)) #contien tout saut le test
    test_indices = []
    for index in range(lr.data_driver.nb_samples_per_mov):
        if index not in training_indices:
            test_indices.append(index)
    data, remains = lr.data_driver.get_data_set(training_indices, shuffle_samples=False)
    nb_training_samples = len(training_indices)

    if show_data:
        lr.show_data_ori(
            sample_indices=slice(nb_training_samples+1, None))

    if leave_one_out:
        # Select the model index on which a leave_one_out was running to resume it
        # if i == 0:
        #     resume = True
        # else:
        #     resume = False
        pass

    if train:
        if leave_one_out:
            # leave_one_out function call for selected models in trainings list
            mse = lr.leave_one_out(double_validation=False, resume=resume)
            #pdb.set_trace()

            print("MSE = "+str(mse))
            print(lr.get_leave_one_out_winner_full_path())
        else:
            lr.train(
                data,
                show_error=False,
                resume=restore,
                save_path=training[0]
            )
    else:
        lr.config.DATA_VISUALIZATION.update(DATA_VISUALIZATION)

        if plot_error:
            lr.plot_error(training[0])

        if show_latent_space:
            zs = []
            lisst = []

            for inx in range(10):
                lisst.append(inx)

            #lr.show_data_ori(sample_indices=[8])
            zs.append(lr.show_latent_space(sample_indices=lisst))
            lr.show_latent_space(sample_indices=[8])
            lr.show_reconstr_data_from_zs(zs, sample_indices=[8])
            lr.show_reconstr_data(sample_indices=[8])
            #for testn in range(7):
                #try:
                    #os.mkdir("./testt/test_"+str(testn))
                #except OSError:
                    #pass
                #for innx in range(10):
                    #f = open("./testt/test_"+str(testn)+"/record"+str(innx)+".txt", "w+")
                    #for vb in range(0,70):
                        #f.write(str(zs[0][0][10*(testn-1)+innx][vb][0])+"\t"+str(zs[0][0][10*(testn-1)+innx][vb][1])+"\n")
                    #f.close()
                    

        lr.config.DATA_VISUALIZATION["displayed_movs"] = reconstr_data_displayed_movs
        #lr.show_inferred_parameters()

        if i == len(trainings)-1:
            if plot_mse:
                lr.plot_mse(
                    compare_to_other_models=True,
                    sample_indices=test_indices
                    # sample_indices=range(nb_training_samples, lr.data_driver.nb_samples_per_mov)
                )

            
            if show_reconstr_data:
                lr.show_reconstr_data(
                    compare_to_other_models=True,
                    sample_indices=test_indices,
                    # sample_indices=range(nb_training_samples, lr.data_driver.nb_samples_per_mov),
                    # sample_indices=range(nb_training_samples),
                    only_hard_joints=True
                )
              
                lr.show_reconstr_data_ori(sample_indices=[8])

                lr.show_reconstr_data(compare_to_other_models=True, sample_indices=None, only_hard_joints=True, average_reconstruction=True)
                
        else:
            lr.init_x_reconstr()

    lrs.append(lr)
    lr.destroy_vtsfe()

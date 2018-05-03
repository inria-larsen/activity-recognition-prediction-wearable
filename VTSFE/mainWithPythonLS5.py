# -*- coding: utf-8 -*-
import numpy as np
import pdb
import os
import yarp
import sys
import pickle
#from launcher import Launcher
from src.launcher import Launcher
from src.connector import Connector
from src.my_statistics import My_statistics


# config files
## joints
#MVNX c'est le format des fichier Xsens MVN
from app.vae_only_joint_mvnx_2D import vae_only_joint_mvnx_2D
from app.vae_only_joint_mvnx_5D import vae_only_joint_mvnx_5D
from app.vae_only_joint_mvnx_7D import vae_only_joint_mvnx_7D
from app.vae_only_position_mvnx_7D import vae_only_position_mvnx_7D

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
#from app.vae_dmp_position_mvnx_2D import vae_dmp_position_mvnx_2D
#from app.vae_dmp_position_mvnx_5D import vae_dmp_position_mvnx_5D
# from app.vae_dmp_joint_mvnx_7D import vae_dmp_joint_mvnx_7D

# from app.tighter_lb_position_mvnx_2D_L15 import tighter_lb_position_mvnx_2D_L15
# from app.tighter_lb_position_mvnx_5D_L15 import tighter_lb_position_mvnx_5D_L15
# from app.tighter_lb_joint_mvnx_7D_L15 import tighter_lb_joint_mvnx_7D_L15

# from app.tighter_lb_joint_mvnx_2D_L30 import tighter_lb_joint_mvnx_2D_L30
# from app.tighter_lb_joint_mvnx_5D_L30 import tighter_lb_joint_mvnx_5D_L30
# from app.tighter_lb_joint_mvnx_7D_L30 import tighter_lb_joint_mvnx_7D_L30

from app.tighter_lb_light_position_mvnx_2D import tighter_lb_light_position_mvnx_2D
#from app.tighter_lb_light_position_mvnx_69D import tighter_lb_light_position_mvnx_69D
# from app.tighter_lb_light_position_mvnx_5D import tighter_lb_light_position_mvnx_5D
# from app.tighter_lb_light_joint_mvnx_7D import tighter_lb_light_joint_mvnx_7D

trainings = []

#######################################################################################################################################################
# EXPERIMENTS   #######################################################################################################################################

# Whether you want to run a leave_one_out in training mode or not
leave_one_out = False
# Select the sample index in [0, 9] on which models will be tested (all other samples will be used for training)
# Given our 7 movement types, this means: 9*7 = 63 training samples, 1*7 = 7 test samples
test_index = 8
if leave_one_out:
    ti = ''
else:
    ti = "_test_"+str(test_index)

#retrieve information about number of epochs used to learn
nb_epochs = 10
for i, val in enumerate(sys.argv):
    if(val == 'nb_epochs'):
        nb_epochs = int(sys.argv[i+1])

print("tighter_lb_light_position_mvnx_2D_nbEpochs_"+str(nb_epochs)+ti)
#trainings.append(("vae_only_joint_mvnx_2D"+ti, vae_only_joint_mvnx_2D))    # OK
# trainings.append(("vae_only_joint_mvnx_5D"+ti, vae_only_joint_mvnx_5D))    # OK
#trainings.append(("vae_only_joint_mvnx_7D"+ti, vae_only_joint_mvnx_7D))    # OK
# trainings.append(("vae_dmp_no_z_derivative_joint_mvnx_2D_separated_encoder_variables", vae_dmp_no_z_derivative_joint_mvnx_2D_separated))  # OK
# trainings.append(("vae_dmp_no_z_derivative_joint_mvnx_2D", vae_dmp_no_z_derivative_joint_mvnx_2D))  # OK
#trainings.append(("vae_dmp_joint_mvnx_2D_separated_encoder_variables"+ti, vae_dmp_joint_mvnx_2D_separated))  # OK
# trainings.append(("vae_dmp_joint_mvnx_5D_separated_encoder_variables"+ti, vae_dmp_joint_mvnx_5D_separated))  # OK
# trainings.append(("vae_dmp_joint_mvnx_7D_separated_encoder_variables"+ti, vae_dmp_joint_mvnx_7D_separated))  # OK
trainings.append(("tighter_lb_light_position_mvnx_2D_nbEpochs_"+str(nb_epochs)+ti, tighter_lb_light_position_mvnx_2D))
#trainings.append(("tighter_lb_light_position_mvnx_69D"+ti, tighter_lb_light_position_mvnx_69D))

#trainings.append(("vae_dmp_position_mvnx_5D"+ti,vae_dmp_position_mvnx_5D ))

#trainings.append(("tighter_lb_light_joint_mvnx_2D_separated_encoder_variables"+ti, tighter_lb_light_joint_mvnx_2D_separated))   # OK
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
nbLS = 2
# show input data space
show_data = False
# plot learning errors through epochs
plot_error = False
# show latent space
show_latent_space = False
# plot the global MSE and MSE for each movement type
plot_mse = False
# show reconstruction of input data space, compare it between models in trainings
show_reconstr_data = True
record_latent_space = False
unitary_tests = False
launch_stats= False
# movement types shown at data reconstruction
reconstr_data_displayed_movs = ["kicking"]#, "bent_fw_strongly"]
commWithMatlab = False
little_stats  = False
restore = False
train = True
lrs = []
mses = []

for i, training in enumerate(trainings):
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
        "nb_epochs": nb_epochs,
        "display_step": 1,
        "checkpoint_step": 10
    })

    training_indices = list(range(test_index))+list(range(test_index+1,10)) #contient tout saut le test
    test_indices = []
    for index in range(lr.data_driver.nb_samples_per_mov):
        if index not in training_indices:
            test_indices.append(index)
    data, remains = lr.data_driver.get_data_set(training_indices, shuffle_samples=False)
    nb_training_samples = len(training_indices)

    if show_data:
        #lr.show_data_ori(
        #   sample_indices=slice(nb_training_samples+1, None))
        lr.show_data(
            sample_indices=[8],#slice(nb_training_samples+1, None),
            only_hard_joints=True
        )

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


        if commWithMatlab:
            
            connex = Connector() #YARP connexion with matlab
            list_zs = []
            for i in range(7):
                connex.addMessage("ask_data")            
                zs = connex.readFloat(nbData = [70,nbLS]) #read latent space that comes from Matlab
                list_zs.append(zs) 
            connex.closeConnector() #YARP disconnexion

            lisst = []
            x_reconstr_from_ls = np.zeros([7,70,69])
            for inx in range(10): # to represent the whole latent space trajectory for all the data set
                lisst.append(inx)

            for i in range(7):
                if(nbLS<15):
                    lr.show_latent_space(list_zs[i], sample_indices=lisst, titleFrame = 'Latent_space_infered for' + str(i))
                x_reconstr_from_ls[i] = lr.retrieve_data_from_latent_space_ori(list_zs[i])
                #lr.show_reconstr_data_ori(x_reconstr_from_ls[i], type_indices={'bent_fw'}, sample_indices=lisst)
                #dataRetrieve = lr.retrieve_data_from_latent_space_ori(zs)

               # lr.show_data(
                    #sample_indices=[8],
                   # only_hard_joints=True,
                  #  data_inf=x_reconstr_from_ls, 
                 #   xreconstr=[]
                #)
            lr.show_reconstr_data(
                compare_to_other_models=True,
                sample_indices=[8],
                # sample_indices=range(nb_training_samples, lr.data_driver.nb_samples_per_mov),
                # sample_indices=range(nb_training_samples),
                only_hard_joints=True,
                data_inf=x_reconstr_from_ls,
            )

        if launch_stats:
            nbStat = 10
            connex = Connector() #YARP connexion with matlab
            global_list = np.zeros([nbStat,70,70,nbLS])
            list_error = np.zeros([nbStat,70])
            #dist_real_reconstr = np.zeros([5,2]) 
            #dist_real_inf = np.zeros([5,2])
            #dist_reconstr_inf = np.zeros([5,2])

            for nbPercent in range(nbStat):
                list_zs = []
                for i in range(70):  
                    connex.addMessage("ask_data")            
                    zs = connex.readFloat(nbData = [70,nbLS]) #read latent space that comes from Matlab
                    list_zs.append(zs) 
                print('ask_error_list')
                connex.addMessage("ask_error_list")
                list_error[nbPercent,:] = connex.readFloat(nbData = [70,1],flag_debug =True);
                global_list[nbPercent] = list_zs
            print("close matlab connexion")
            connex.closeConnector() #YARP disconnexion

            x_reconstr_from_ls = np.zeros([nbStat,70,70,69])
            for nbPercent in range(nbStat):
                sample_indices = []

                for i in range(70):
                    x_reconstr_from_ls[nbPercent,i] = lr.retrieve_data_from_latent_space_ori(global_list[nbPercent,i])

                    if(list_error[nbPercent,i]==0):
                        sample_indices.append(i+1)
            
            my_statistics = lr.compute_stats(compare_to_other_models=True, nb_samples_per_mov=10, sample_indices=sample_indices, only_hard_joints=True, data_inf=x_reconstr_from_ls)
            
            with open('myLongStats_LS'+nbLS, 'wb') as fichier:
                mon_pickler= pickle.Pickler(fichier)
                mon_pickler.dump(my_statistics)
            #dist_real_reconstr[nbPercent,:], dist_real_inf[nbPercent,:], dist_reconstr_inf[nbPercent,:] = lr.compute_stats(compare_to_other_models=True, nb_samples_per_mov=10, sample_indices=sample_indices, only_hard_joints=True, data_inf=x_reconstr_from_ls[nbPercent])
            #lr.show_reconstr_data(compare_to_other_models=True, nb_samples_per_mov=3, only_hard_joints=True, data_inf=x_reconstr_from_ls)#sample_indices=[8],
                # sample_indices=range(nb_training_samples, lr.data_driver.nb_samples_per_mov),
                # sample_indices=range(nb_training_samples),

        if little_stats:
            my_statistics = lr.compute_stats(compare_to_other_models=True, nb_samples_per_mov=10, sample_indices=[8], only_hard_joints=True, data_inf=[])
        
            with open('myLittleStats_LS2_epochs_'+str(nb_epochs), 'wb') as fichier:
                mon_pickler= pickle.Pickler(fichier)
                mon_pickler.dump(my_statistics)

        

        if unitary_tests:
            """Test if the program can retrieve the laten space, plot it correctly, reconstruct one latent space correctly and plot it correctly"""
            lisst = []
            for inx in range(10):
                lisst.append(inx)
            
            zs = np.array(lr.show_latent_space(sample_indices=lisst))
            #for inx in range(7):
            test = zs[0, :, 0, :] #test sample_indice = 0 bent_fw
            lr.show_latent_space(test, sample_indices=lisst)

            #Test pour verifier si on peut recuperer xrecovered depuis zs et le ploter
            x_reconstr_from_ls = lr.retrieve_data_from_latent_space_ori(test)
            lr.show_reconstr_data_ori(x_reconstr_from_ls, type_indices={'bent_fw'}, sample_indices=[8])

            lr.show_reconstr_data(
                    compare_to_other_models=True,
                    sample_indices=test_indices,
                    # sample_indices=range(nb_training_samples, lr.data_driver.nb_samples_per_mov),
                    # sample_indices=range(nb_training_samples),
                    only_hard_joints=True,
                    data_inf=x_reconstr_from_ls,
            )


            #test Unitaire : avec latent space pas correct (pour verif reconstr utlise bien le latent space test)
            for idx in range(70):
                test[idx,0] += 3
                test[idx,1] -= 3
            lr.show_latent_space(test, sample_indices=lisst)
            x_reconstr_from_ls = lr.retrieve_data_from_latent_space_ori(test)
            lr.show_reconstr_data_ori(x_reconstr_from_ls, type_indices={'bent_fw'}, sample_indices=[8])

            lr.show_reconstr_data(
                    compare_to_other_models=True,
                    sample_indices=test_indices,
                    # sample_indices=range(nb_training_samples, lr.data_driver.nb_samples_per_mov),
                    # sample_indices=range(nb_training_samples),
                    only_hard_joints=True,
                    data_inf=x_reconstr_from_ls,
            )


        if show_latent_space:
            lisst = []
            for inx in range(10):
                lisst.append(inx)

            zs = np.array(lr.show_latent_space(sample_indices=lisst))

            #Si l'on veut sauvegarder les données de l'espace latent
        if record_latent_space:
            lisst = []
            for inx in range(10):
                lisst.append(inx)

            zs = np.array(lr.compute_latent_space(sample_indices=lisst))

            for testn in range(7):
                try:
                    os.mkdir("./testt/l69_"+str(testn))
                except OSError:
                    pass
                for innx in range(10):
                    f = open("./testt/l69_"+str(testn)+"/record"+str(innx)+".txt", "w+")
                    for vb in range(0,70):
                        nameString = ''
                        for nbstring in range(nbLS-1):
                            nameString += str(zs[0,vb,10*(testn)+innx,nbstring])+"\t"
                        nameString +=str(zs[0,vb,10*(testn)+innx,nbLS-1])+"\n"
                
                        f.write(nameString)
                    f.close()
                    

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
                #lr.show_reconstr_data(
                    #compare_to_other_models=True,
                    #sample_indices=test_indices,
                    ## sample_indices=range(nb_training_samples, lr.data_driver.nb_samples_per_mov),
                    ## sample_indices=range(nb_training_samples),
                    #only_hard_joints=True
                #)

#                lr.show_reconstr_data_ori(sample_indices=[8])

                lr.show_reconstr_data(
                    compare_to_other_models=True, 
                    sample_indices=None, 
                    only_hard_joints=True, 
                   average_reconstruction=True)
                
        else:
            lr.init_x_reconstr()
    lrs.append(lr)
    lr.destroy_vtsfe()

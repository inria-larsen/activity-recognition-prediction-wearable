from .models.tighter_lb_light_2D import Tighter_LB_Light_2D

tighter_lb_light_joint_mvnx_7D_separated = Tighter_LB_Light_2D()
tighter_lb_light_joint_mvnx_7D_separated.set_data_config(["jointAngle"], "MVNX")
tighter_lb_light_joint_mvnx_7D_separated.VTSFE_PARAMS["separate_losses"] = True

tighter_lb_light_joint_mvnx_7D_separated.VTSFE_PARAMS["vae_architecture"]["n_z"] = 7

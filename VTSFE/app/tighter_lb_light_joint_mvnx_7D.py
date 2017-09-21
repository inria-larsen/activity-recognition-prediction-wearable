from .models.tighter_lb_light_2D import Tighter_LB_Light_2D

tighter_lb_light_joint_mvnx_7D = Tighter_LB_Light_2D()
tighter_lb_light_joint_mvnx_7D.set_data_config(["jointAngle"], "MVNX")

tighter_lb_light_joint_mvnx_7D.VTSFE_PARAMS["vae_architecture"]["n_z"] = 7

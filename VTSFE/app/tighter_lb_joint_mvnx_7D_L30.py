from .models.tighter_lb_2D import Tighter_LB_2D

tighter_lb_joint_mvnx_7D_L30 = Tighter_LB_2D()
tighter_lb_joint_mvnx_7D_L30.set_data_config(["jointAngle"], "MVNX")

tighter_lb_joint_mvnx_7D_L30.VTSFE_PARAMS["vae_architecture"]["n_z"] = 7

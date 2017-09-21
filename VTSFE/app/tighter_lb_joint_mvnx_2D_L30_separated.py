from .models.tighter_lb_2D import Tighter_LB_2D

tighter_lb_joint_mvnx_2D_L30_separated = Tighter_LB_2D()
tighter_lb_joint_mvnx_2D_L30_separated.set_data_config(["jointAngle"], "MVNX")

tighter_lb_joint_mvnx_2D_L30_separated.VTSFE_PARAMS["separate_losses"] = True

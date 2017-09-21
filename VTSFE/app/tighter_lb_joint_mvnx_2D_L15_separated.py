from .models.tighter_lb_2D import Tighter_LB_2D

tighter_lb_joint_mvnx_2D_L15_separated = Tighter_LB_2D()
tighter_lb_joint_mvnx_2D_L15_separated.set_data_config(["jointAngle"], "MVNX")

tighter_lb_joint_mvnx_2D_L15_separated.DMP_PARAMS["model_monte_carlo_sampling"] = 15
tighter_lb_joint_mvnx_2D_L15_separated.VTSFE_PARAMS["separate_losses"] = True

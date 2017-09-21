from .models.tighter_lb_light_2D import Tighter_LB_Light_2D

tighter_lb_light_joint_mvnx_2D_separated = Tighter_LB_Light_2D()
tighter_lb_light_joint_mvnx_2D_separated.set_data_config(["jointAngle"], "MVNX")
tighter_lb_light_joint_mvnx_2D_separated.VTSFE_PARAMS["separate_losses"] = True

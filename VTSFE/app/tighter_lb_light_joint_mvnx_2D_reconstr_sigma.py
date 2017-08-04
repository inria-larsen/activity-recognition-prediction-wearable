from .models.data import get_data_config
from .models.tighter_lb_light_2D import Tighter_LB_Light_2D

tighter_lb_light_joint_mvnx_2D_reconstr_sigma = Tighter_LB_Light_2D()
tighter_lb_light_joint_mvnx_2D_reconstr_sigma.DATA_PARAMS = get_data_config(["jointAngle"], "MVNX")

tighter_lb_light_joint_mvnx_2D_reconstr_sigma.VTSFE_PARAMS["vae_architecture"]["use_reconstr_sigma"] = True

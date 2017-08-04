from .models.data import get_data_config
from .models.tighter_lb_light_2D import Tighter_LB_Light_2D

tighter_lb_light_joint_mvnx_5D = Tighter_LB_Light_2D()
tighter_lb_light_joint_mvnx_5D.DATA_PARAMS = get_data_config(["jointAngle"], "MVNX")

tighter_lb_light_joint_mvnx_5D.VTSFE_PARAMS["vae_architecture"]["n_z"] = 5

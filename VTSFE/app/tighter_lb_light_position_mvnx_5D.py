from .models.tighter_lb_light_2D import Tighter_LB_Light_2D

tighter_lb_light_position_mvnx_5D = Tighter_LB_Light_2D()
tighter_lb_light_position_mvnx_5D.set_data_config(["position"], "MVNX")

tighter_lb_light_position_mvnx_5D.VTSFE_PARAMS["vae_architecture"]["n_z"] = 5

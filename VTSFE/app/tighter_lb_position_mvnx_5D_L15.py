from .models.tighter_lb_2D import Tighter_LB_2D

tighter_lb_position_mvnx_5D_L15 = Tighter_LB_2D()
tighter_lb_position_mvnx_5D_L15.set_data_config(["position"], "MVNX")

tighter_lb_position_mvnx_5D_L15.VTSFE_PARAMS["vae_architecture"]["n_z"] = 5
tighter_lb_position_mvnx_5D_L15.DMP_PARAMS["model_monte_carlo_sampling"] = 15

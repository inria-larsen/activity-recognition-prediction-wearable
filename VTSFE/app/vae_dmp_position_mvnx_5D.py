from .models.vae_dmp_2D import VAE_DMP_2D

vae_dmp_position_mvnx_5D = VAE_DMP_2D()
vae_dmp_position_mvnx_5D.set_data_config(["position"], "MVNX")

vae_dmp_position_mvnx_5D.VTSFE_PARAMS["vae_architecture"]["n_z"] = 5

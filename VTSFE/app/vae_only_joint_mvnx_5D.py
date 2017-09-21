from .models.vae_only_2D import VAE_Only_2D

vae_only_joint_mvnx_5D = VAE_Only_2D()
vae_only_joint_mvnx_5D.set_data_config(["jointAngle"], "MVNX")

vae_only_joint_mvnx_5D.VTSFE_PARAMS["vae_architecture"]["n_z"] = 5

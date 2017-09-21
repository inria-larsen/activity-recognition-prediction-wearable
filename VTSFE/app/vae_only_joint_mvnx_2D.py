from .models.vae_only_2D import VAE_Only_2D

vae_only_joint_mvnx_2D = VAE_Only_2D()
vae_only_joint_mvnx_2D.set_data_config(["jointAngle"], "MVNX")

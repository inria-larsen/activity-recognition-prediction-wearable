from .models.vae_dmp_2D import VAE_DMP_2D

vae_dmp_joint_mvnx_7D = VAE_DMP_2D()
vae_dmp_joint_mvnx_7D.set_data_config(["jointAngle"], "MVNX")

vae_dmp_joint_mvnx_7D.VTSFE_PARAMS["vae_architecture"]["n_z"] = 7

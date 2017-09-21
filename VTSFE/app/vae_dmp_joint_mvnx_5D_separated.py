from .models.vae_dmp_2D import VAE_DMP_2D

vae_dmp_joint_mvnx_5D_separated = VAE_DMP_2D()
vae_dmp_joint_mvnx_5D_separated.set_data_config(["jointAngle"], "MVNX")
vae_dmp_joint_mvnx_5D_separated.VTSFE_PARAMS["separate_losses"] = True

vae_dmp_joint_mvnx_5D_separated.VTSFE_PARAMS["vae_architecture"]["n_z"] = 5

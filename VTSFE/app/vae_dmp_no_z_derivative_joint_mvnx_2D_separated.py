from .models.vae_dmp_no_z_derivative_2D import VAE_DMP_No_Z_Derivative_2D

vae_dmp_no_z_derivative_joint_mvnx_2D_separated = VAE_DMP_No_Z_Derivative_2D()
vae_dmp_no_z_derivative_joint_mvnx_2D_separated.set_data_config(["jointAngle"], "MVNX")

vae_dmp_no_z_derivative_joint_mvnx_2D_separated.VTSFE_PARAMS["separate_losses"] = True

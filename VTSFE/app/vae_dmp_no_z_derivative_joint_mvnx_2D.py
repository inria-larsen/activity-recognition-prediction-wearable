from .models.vae_dmp_no_z_derivative_2D import VAE_DMP_No_Z_Derivative_2D

vae_dmp_no_z_derivative_joint_mvnx_2D = VAE_DMP_No_Z_Derivative_2D()
vae_dmp_no_z_derivative_joint_mvnx_2D.set_data_config(["jointAngle"], "MVNX")

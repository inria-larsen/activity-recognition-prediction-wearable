from .models.data import get_data_config
from .models.vae_dmp_2D import VAE_DMP_2D

vae_dmp_joint_mvnx_2D = VAE_DMP_2D()
vae_dmp_joint_mvnx_2D.DATA_PARAMS = get_data_config(["jointAngle"], "MVNX")

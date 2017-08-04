from .models.data import get_data_config
from .models.tighter_lb_2D import Tighter_LB_2D

tighter_lb_joint_mvnx_2D_L30 = Tighter_LB_2D()
tighter_lb_joint_mvnx_2D_L30.DATA_PARAMS = get_data_config(["jointAngle"], "MVNX")

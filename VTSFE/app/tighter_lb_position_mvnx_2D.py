from .models.data import get_data_config
from .models.tighter_lb_2D import Tighter_LB_2D

tighter_lb_position_mvnx_2D = Tighter_LB_2D()
tighter_lb_position_mvnx_2D.DATA_PARAMS = get_data_config(["position"], "MVNX")

from hmm_model import ModelHMM
from data_base import DataBase
import data_processing as pr
import numpy as np
import matplotlib.pyplot as plt
import sys
import yarp
import visualization_tools as v_tools
import tools
import pandas as pd 


data_win = []
labels = []

for i in range(40):
	data_win.append(np.zeros((10,30)))
	labels.append('aaa')

ratio = 70

data_ref, labels_ref, data_test, labels_test = tools.split_data_base(data_win, labels, ratio)
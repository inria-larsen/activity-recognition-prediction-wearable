import numpy as np
import os
from eglove_parser import eglove_tree
import pandas as pd


path = '/home/amalaise/Documents/Xsens/data/pick_and_place_carry/0/'

list_files = os.listdir(path + 'eglove')
list_files.sort()

eglove_files = []

for file in list_files:
	tree = eglove_tree(path + 'eglove/' + file)
	data = tree.get_all_data()
	time = tree.get_timestamp()
	time = np.expand_dims(time, axis=1)

	data = np.concatenate((data, time), axis = 1)

	df = pd.DataFrame(data)
	df.to_csv(path + 'glove/' + file + '.csv', index = False, header=False, sep=' ')

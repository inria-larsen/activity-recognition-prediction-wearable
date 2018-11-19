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
from copy import deepcopy
from mem_top import mem_top
from sys import getsizeof
import os
import re
from scipy import stats

import warnings
warnings.filterwarnings("ignore")




if __name__ == '__main__':
	# Prepare the data base
	yarp.Network.init()
	rf = yarp.ResourceFinder()
	rf.setDefaultContext("online_recognition")
	rf.setDefaultConfigFile("default.ini")
	rf.configure(sys.argv)

	path = rf.find('path_data_base').toString()
	path_model = rf.find('path_model').toString()
	name_model = rf.find('name_model').toString()
	name_track = rf.find('level_taxonomy').toString()
	labels_folder = rf.find('labels_folder').toString()

	list_participant = os.listdir(path)
	list_participant.sort()
	print(list_participant)

	# list_participant = ['5124']
	participant_label = []
	num_sequence = []
	testing = 1
	save = 0
	ratio = [70, 30, 0]
	nbr_cross_val = 10
	test_generalisation = 0
	method = 'wrapper'
	test_iteration = 1

	nbr_features = 50
	

	list_participant = ['541', '909', '3327', '5124', '5521', '5535', '8410', '9266', '9875']
	# list_participant = ['541']

	# # path_video = '/home/amalaise/Documents/These/experiments/AnDy-LoadHandling/annotation/Videos_Xsens/Participant_541/Participant_541_Setup_A_Seq_3_Trial_1.mp4'


	# id_test = 0

	print('Loading data...')

	timestamps = []
	data_win2 = []
	real_labels = []
	list_states = []

	info_participant = []
	info_sequences = []
	i=0

	for participant, nbr in zip(list_participant, range(len(list_participant))):
		path_data = path + '/' + participant + '/data_csv/'
		print('Loading: ' + participant)
		
		list_files = os.listdir(path_data)
		list_files.sort()

		for file in list_files:
			name_seq = os.path.splitext(file)[0]

			info_participant.append(participant)
			info_sequences.append(name_seq)

			data_base = pd.read_csv(path_data + file)
			ref_data = DataBase(path + '/' + participant, name_seq)

# 		data, time, list_features, dim_features = tools.load_data_from_dataBase(data_base, rf)
# 		for d, t in zip(data, time):
# 			data_win2.append(d)
# 			timestamps.append(t)

			list_features = list(data_base.columns.values)
			del list_features[0:2]
			dim_features = np.ones(len(list_features))

			time = data_base['timestamps']

			
			labels, states = ref_data.load_labels_refGT(time, name_track, 'labels_3A')
			# ref_data.load_labels_ref(name_track, labels_folder)
			# labels = ref_data.get_real_labels(time)
			# states = ref_data.get_list_states()

			real_labels.append(labels)
			data_win2.append(data_base[list_features].as_matrix())
			timestamps.append(time)

			

			for state in states:
				if(state not in list_states):
					list_states.append(state)
					list_states = sorted(list_states)
			i += 1

	print(list_states)


	obs = []
	obs = data_win2[0]
	lengths = []
	lengths.append(len(data_win2[0]))
	labels = real_labels[0]

	for i in range(1, len(data_win2)):
		obs = np.concatenate([obs, data_win2[i]])
		lengths.append(len(data_win2[i]))
		labels = np.concatenate([labels, real_labels[i]])


	df_data = pd.DataFrame(obs, columns = list_features)
	df_labels = pd.DataFrame(labels)
	df_labels.columns = ['state']

	df_total = pd.concat([df_data, df_labels], axis=1)

	data = []

	
	for state, df in df_total.groupby('state'):
		print('\n', state)
		for features in list_features:
			print(features, stats.shapiro(df[features]))

		# data.append(df[list_features].values)

	# hist = df_data['jointAngleNorm_jL5S1_0'].hist
	# ax = hist.plot.hist(bins=12, alpha=0.5)
		fig, ax = plt.subplots()
		df['CoMNormalize_2'].plot.hist(label=state, bins=100, alpha=0.5, ax=ax)
		ax.legend()
		# # plt.title(state)

	plt.show()
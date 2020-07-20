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
import pickle
import pylab 

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
	# Prepare the data base
	yarp.Network.init()
	rf = yarp.ResourceFinder()
	rf.setDefaultContext("online_recognition")
	rf.setDefaultConfigFile("general_posture.ini")
	rf.configure(sys.argv)

	path = rf.find('path_data_base').toString()
	path_model = rf.find('path_model').toString()
	name_model = rf.find('name_model').toString()
	name_track = rf.find('level_taxonomy').toString()
	labels_folder = rf.find('labels_folder').toString()

	path_data_root = path + '/xsens/allFeatures_csv/'
	path_wrapper = '/home/amalaise/Documents/These/experiments/ANDY_DATASET/AndyData-lab-onePerson/'


	list_participant = os.listdir(path_data_root)
	list_participant.sort()

	print(list_participant)

	num_sequence = []
	testing = 1
	save = 0
	ratio = [70, 30, 0]
	nbr_cross_val = 10

	nbr_features = 50

	list_participant = ['Participant_541']


	print('Loading data...')

	data_win2 = []
	real_labels = [[],[],[],[]]
	list_states = [[], [], [], []]


	tracks = ['general_posture', 'detailed_posture', 'details', 'current_action']

	path_annotation = '/home/amalaise/Documents/These/experiments/ANDY_DATASET/AndyData-lab-onePerson/annotations/labels_csv2/'


	# with open('score/save_data_dump.pkl', 'rb') as input:
	# 	data_win2 = pickle.load(input)
	# with open('score/save_labels_dump.pkl', 'rb') as input:
	# 	real_labels = pickle.load(input)
	with open('score/save_liststates_dump.pkl', 'rb') as input:
		list_states = pickle.load(input)
	with open('score/save_listfeatures_dump.pkl', 'rb') as input:
		list_features = pickle.load(input)

	with open('score/save_dfdata_dump.pkl', 'rb') as input:
		df_total = pickle.load(input)

	# i = 0
	# for participant, nbr in zip(list_participant, range(len(list_participant))):
	# 	path_data = path_data_root  + participant
	# 	print('Loading: ' + participant)
		
	# 	list_files = os.listdir(path_data)[0:1]
	# 	list_files.sort()

	# 	for file in list_files:
	# 		name_seq = os.path.splitext(file)[0]
			
	# 		# data, time, list_features = tools.load_data(path, participant, name_seq, 'general_posture', labels_folder)
	# 		data, labels, time, list_s, list_features = tools.load_data(path, participant, name_seq, 'general_posture', labels_folder)

	# 		data_win2.append(data)

	# 		for name_track, num_track in zip(tracks, range(len(tracks))):
	# 			labels, states = tools.load_labels_ref(time, path_annotation + participant + '/' + name_seq + '.labels.csv',
	# 				name_track, participant, 1)
	# 			real_labels[num_track].append(labels)

	# 			for state in states:
	# 				if(state not in list_states[num_track]):
	# 					list_states[num_track].append(state)
	# 					list_states[num_track] = sorted(list_states[num_track])
	# 		i += 1

	# print(list_states)

	# obs = []
	# labels = [[],[],[],[]]

	# for i in range(0, len(data_win2)):
	# 	if(i==0):
	# 		obs = data_win2[0]
	# 		for k in range(len(tracks)):
	# 			labels[k] = real_labels[k][0]
	# 	else:
	# 		obs = np.concatenate([obs, data_win2[i]])
	# 		for k in range(len(tracks)):
	# 			labels[k] = np.concatenate([labels[k], real_labels[k][i]])

	# df_data = pd.DataFrame(obs, columns = list_features)
	# df_labels = pd.DataFrame(labels).T
	# df_labels.columns = tracks

	# df_total = pd.concat([df_data, df_labels], axis=1)

	# with open('score/save_dfdata_dump.pkl', 'wb') as output:
	# 	pickle.dump(df_total, output, pickle.HIGHEST_PROTOCOL)


	# df_total.to_csv('score/all_data.csv', index=False)

	# data = []

	# shapiro_value = []
	# all_features = []
	# all_states = []
	# for k in range(len(tracks))
	# 	for state, df in df_total.groupby(tracks[k]):
	# 		for features in list_features:
	# 			shapiro_value.append(stats.shapiro(df[features])[1])
	# 			all_features.append(features)
	# 			all_states.append(state)

	# df_norm_test = pd.DataFrame({'state': all_states,
	# 		'features': all_features,
	# 		'p-value': shapiro_value})

	# df_norm_test = df_norm_test.sort_values(by=['p-value'], ascending=False)
	# df_norm_test.to_csv('score/gaussian_test_' + name_track + ".csv", index=False)

	f = 'comPos_centerOfMass_z'
	track = 'detailed_posture'

	data_test = df_total[[f, track]]

	# data.append(df[list_features].values)

	# hist = data_test.hist
	# ax = hist.plot.hist(bins=12, alpha=0.5)
	
	for state, df in data_test.groupby(track):
		print(state, df.mean(), df.std())
		fig, ax = plt.subplots()
		stats.probplot(df[f], dist="norm", plot=pylab)
		# 
		# # df['velocityNorm'].plot.hist(label=state, bins=100, alpha=0.5, ax=ax)
		ax.legend()
		plt.title('Center of Mass position (z) QQ Plot - ' + state)

		fig.savefig("/home/amalaise/Documents/These/papers/adrien_ra-l/img/" + state + ".pdf", bbox_inches='tight')

	# measurements = np.random.normal(loc = 20, scale = 5, size=100)   

	pylab.show()

	plt.show()
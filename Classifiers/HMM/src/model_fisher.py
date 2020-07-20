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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

	path_data_root = path + '/xsens/allFeatures_csv/'


	list_participant = os.listdir(path_data_root)
	list_participant.sort()

	print(list_participant)


	# list_participant = ['Participant_5124']
	participant_label = []
	num_sequence = []
	testing = 1
	save = 0
	ratio = [70, 30, 0]
	nbr_cross_val = 10
	test_generalisation = 0
	method = 'wrapper'
	test_iteration = 1
	local_features = 1

	nbr_features = 50

	nbr_component = 15
	
	print('Loading data...')

	data_win2 = []
	real_labels = [[],[],[],[]]
	list_states = [[], [], [], []]

	tracks = ['general_posture', 'detailed_posture', 'details', 'current_action']

	path_annotation = '/home/amalaise/Documents/These/experiments/ANDY_DATASET/AndyData-lab-onePerson/annotations/labels_csv2/'


	data_win2, real_labels, list_states, list_features = tools.load_data_from_dump('score/')
	if(local_features):
		list_reduce_features = tools.list_features_local(list_features)
		for data in data_win2:
			df_data  = pd.DataFrame(data, columns = list_features)
			data = df_data[list_reduce_features].values



	# for participant, nbr in zip(list_participant, range(len(list_participant))):
	# 	path_data = path_data_root  + participant
	# 	print('Loading: ' + participant)
		
	# 	list_files = os.listdir(path_data)[0:3]
	# 	list_files.sort()


	# 	for file in list_files:
	# 		name_seq = os.path.splitext(file)[0]
			
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

	print(list_states)

	for name_track, num_track in zip(tracks, range(len(tracks))):
		print(name_track)
		F1_score = []
		dim_score = []
		feaures_save = []

		for n_components in range(1, nbr_component + 1):
			for n_iter in range(nbr_cross_val):

				
				data_ref, labels_ref, data_test, labels_test, id_train, id_test = tools.split_data_base2(data_win2, real_labels[num_track], ratio)

				obs = data_ref[0]
				labels = labels_ref[0]

				for i in range(1, len(data_ref)):
					obs = np.concatenate([obs, data_ref[i]])
					labels = np.concatenate([labels, labels_ref[i]])


				list_s, labels = np.unique(labels, return_inverse=True)

				df_data = pd.DataFrame(obs, columns = list_features)

				df_labels = pd.DataFrame(labels)
				df_labels.columns = ['state']

				df_total = pd.concat([df_data, df_labels], axis=1)

				data = []
				for state, df in df_total.groupby('state'):
					data.append(df[list_features].values)

				f_score = tools.fisher_score(data, obs)
				list_id_sort = np.argsort(f_score).tolist()
				sorted_score = []
				sorted_features_fisher = []

				for id_sort in reversed(list_id_sort):
					sorted_features_fisher.append(list_features[id_sort])
					sorted_score.append(f_score[id_sort])

				score_totaux = pd.DataFrame(
					{'best_features': sorted_features_fisher,
					 'score': sorted_score,
					})

				best_features = sorted_features_fisher[0:n_components]
				dim_features = np.ones(len(best_features))

				# print(sorted_features_fisher)



				data_reduce = []
				for data in data_ref:
					df = pd.DataFrame(data)
					df.columns = list_features
					data_reduce.append(df[best_features].values)

				model = ModelHMM()
				model.train(data_reduce, labels_ref, best_features, dim_features)


				data_reduce = []
				for data in data_test:
					df = pd.DataFrame(data)
					df.columns = list_features
					data_reduce.append(df[best_features].values)

				pred_labels, proba = model.test_model(data_reduce)

				F1_temp = []
				for i in range(len(labels_test)):
					F1_temp.append(tools.compute_F1_score(labels_test[i], pred_labels[i], list_states[num_track]))

				F1_score.append(np.mean(F1_temp))
				dim_score.append(n_components)
				feaures_save.append(str(best_features))

			score_totaux = pd.DataFrame(
			{'nbr_components': dim_score,
			 'score': F1_score,
			 'features': feaures_save
			})

			score_totaux.to_csv('score/score_fisher' + '_' + name_track + "_local.csv", index=False)





	# plt.show()
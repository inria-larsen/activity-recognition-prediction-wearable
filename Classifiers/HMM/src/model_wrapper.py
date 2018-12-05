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

def get_best_features(file_name):
	best_features = []
	df = pd.read_csv(file_name)
	for i in range(len(df)):
		line = df['best_features'].values[i] # Find the set of features with the best score
		line = line.replace(',', '')
		line = line.replace("[","")
		line = line.replace("]","")
		line = line.replace("'","")
		best_features.append(line.split())
	return best_features


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

	# list_participant = ['Participant_541']


	print('Loading data...')

	data_win2 = []
	real_labels = [[],[],[],[]]
	list_states = [[], [], [], []]

	tracks = ['general_posture', 'detailed_posture', 'details', 'current_action']

	path_annotation = '/home/amalaise/Documents/These/experiments/ANDY_DATASET/AndyData-lab-onePerson/annotations/labels_csv2/'

	for participant, nbr in zip(list_participant, range(len(list_participant))):
		path_data = path_data_root  + participant
		print('Loading: ' + participant)
		
		list_files = os.listdir(path_data)[0:4]
		list_files.sort()


		for file in list_files:
			name_seq = os.path.splitext(file)[0]
			
			data, labels, time, list_s, list_features = tools.load_data(path, participant, name_seq, 'general_posture', labels_folder)
			data_win2.append(data)

			for name_track, num_track in zip(tracks, range(len(tracks))):
				labels, states = tools.load_labels_ref(time, path_annotation + participant + '/' + name_seq + '.labels.csv',
					name_track, participant, 1)
				real_labels[num_track].append(labels)

				for state in states:
					if(state not in list_states[num_track]):
						list_states[num_track].append(state)
						list_states[num_track] = sorted(list_states[num_track])
			

	print(list_states)

	for name_track, num_track in zip(tracks, range(len(tracks))):

		F1_score = []
		dim_score = []
		feaures_save = []
		
		dim = 11
		file_wrapper = path_wrapper + 'wrapper_' + name_track + ".csv_" + str(dim)
		# for dim in range(1, 11):
		while(os.path.isfile(file_wrapper)):
			print(file_wrapper)
			list_features_final = get_best_features(file_wrapper)[0]
			dim_features = np.ones(len(list_features_final))

			for n_iter in range(nbr_cross_val):
				data_ref1, labels_ref, data_test, labels_test, id_train, id_test = tools.split_data_base2(data_win2, real_labels[num_track], ratio)

				data_ref = []
				for data in data_ref1:
					df = pd.DataFrame(data)
					df.columns = list_features
					data_ref.append(df[list_features_final].values)

				model = ModelHMM()
				model.train(data_ref, labels_ref, list_features_final, dim_features)

				data_ref = []
				for data in data_test:
					df = pd.DataFrame(data)
					df.columns = list_features
					data_ref.append(df[list_features_final].values)

				pred_labels, proba = model.test_model(data_ref)

				F1_temp = []
				for i in range(len(labels_test)):
					F1_temp.append(tools.compute_F1_score(labels_test[i], pred_labels[i], list_states[num_track]))

				F1_score.append(np.mean(F1_temp))
				dim_score.append(dim)
				feaures_save.append(str(list_features_final))
				
			score_totaux = pd.DataFrame(
			{'nbr_components': dim_score,
			 'score': F1_score,
			 'features': feaures_save
			})

			score_totaux.to_csv('score/score_model_wrapper_' + name_track + "2.csv", index=False)
			dim += 1
			file_wrapper = path_wrapper + 'wrapper_' + name_track + ".csv_" + str(dim)




	# plt.show()
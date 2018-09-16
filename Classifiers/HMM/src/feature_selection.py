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
import time as TIME_

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

	max_iter = 10

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
	list_participant = ['541', '909', '3327', '5124', '5521', '5535', '8410', '9266', '9875']
	print(list_participant)

	# list_participant = ['909']

	participant_label = []
	num_sequence = []
	testing = 1
	save = 0
	ratio = [70, 30, 0]
	nbr_cross_val = 3

	print('Loading data...')

	timestamps = []

	real_labels = []
	list_states = []

	df_all_data = []

	i=0
	for participant, nbr in zip(list_participant, range(len(list_participant))):
		path_data = path + '/' + participant + '/data_csv/'
		print('Loading: ' + participant)
		
		list_files = os.listdir(path_data)
		list_files.sort()

		for file in list_files:
			name_seq = os.path.splitext(file)[0]

			data, labels, time, list_s, list_features = tools.load_data(path + '/' + participant + '/', name_seq, name_track, labels_folder)
			real_labels.append(labels)



			df_all_data.append(pd.DataFrame(data, columns = list_features))


			timestamps.append(time)

			for state in list_s:
				if(state not in list_states):
					list_states.append(state)
					list_states = sorted(list_states)
			i += 1

	print(list_states)
	dim_features = np.ones(len(list_features))




	print('Data Loaded')

	score = []
	best_features = []
	flag = 0

	# name_track = 'details'
	file_name = 'wrapper_' + name_track + '.csv'
	# file_name = 'wrapper_' + 'details' + ".csv"
	if(os.path.isfile(path + '/' + file_name)):
		best_features = get_best_features(path + '/' + file_name)
		score = pd.read_csv(path + '/' + file_name)['score'].values.tolist()
		file_name = 'wrapper_' + name_track + "1.csv"
		flag = 1

	print(np.shape(score), np.shape(best_features))


	for iteration in range(max_iter):
		
		print('\n#############################')
		print('Iteration: ' + str(iteration))
		print('#############################')
		tic = TIME_.clock()

		if(iteration >= 1 or flag == 1):
			if(len(best_features) >= 10):
				top_list_feature = [deepcopy(best_features[0])]
			else:
				top_list_feature = deepcopy(best_features[0:-1])

		else:
			top_list_feature = ['']

		for top_feature in top_list_feature:
			count = 0

			print('##')
			print(top_feature)
			print('##')

			for feature in list_features:
				count += 1
				if(count % 100 == 0):
					print(count/len(list_features)*100)

				if(iteration > 0 or flag == 1):
					if(feature in top_feature):
						continue
					sub_list_features = deepcopy(top_feature)
					sub_list_features.append(feature)
					sub_list_features = sorted(sub_list_features)

					if(sub_list_features in top_feature):
						continue

				else:
					sub_list_features = [feature]

		###################################################
				
				data_win = []
				
				for j in range(len(df_all_data)):
					data_win.append(df_all_data[j][sub_list_features].values)
				
				F1_S = 0
				MCC = 0

				confusion_matrix = np.zeros((len(list_states), len(list_states)))
				for nbr_test in range(nbr_cross_val):
					
					data_ref, labels_ref, data_test, labels_test, id_train, id_test = tools.split_data_base2(data_win, real_labels, ratio)

					model = ModelHMM()
					model.train(data_ref, labels_ref, sub_list_features, np.ones(len(sub_list_features)))

					#### Test
					time_test = []
					for id_subject in id_test:
						time_test.append(timestamps[id_subject])

					predict_labels, proba = model.test_model(data_test)

					for i in range(len(predict_labels)):
						conf_mat = tools.compute_confusion_matrix(predict_labels[i], labels_test[i], list_states)
						confusion_matrix += conf_mat
						MCC += tools.compute_MCC_score(predict_labels[i], labels_test[i], list_states)/len(predict_labels)



				prec_total, recall_total, F1_score = tools.compute_score(confusion_matrix)	
				acc = tools.get_accuracy(confusion_matrix)
				F1_S = F1_score
					# F1_S = MCC/nbr_cross_val
				if(len(score) == 0):
					score.append(F1_S)
					best_features.append(sub_list_features)
				else:

					for num in range(len(score)):
						if(F1_S > score[num]):
							score.insert(num, F1_S)
							best_features.insert(num, sub_list_features)
							break

						if(num == len(score)-1):
							score.append(F1_S)
							best_features.append(sub_list_features)

			score_totaux = pd.DataFrame(
				{'best_features': best_features,
				 'score': score,
				})
			# score_totaux.to_csv(path + '/' + file_name, index=False)

			toc = TIME_.clock()
			print('Time: ', toc - tic)

	plt.show()



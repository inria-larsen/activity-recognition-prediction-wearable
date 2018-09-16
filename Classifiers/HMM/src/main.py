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

import warnings
warnings.filterwarnings("ignore")

def find_best_features(file_name):
	df = pd.read_csv(file_name)
	line = df['features'].values[df['score'].idxmax()] # Find the set of features with the best score
	line = re.sub(',', '', line)
	line = line.replace("[","")
	line = line.replace("]","")
	line = line.replace("'","")
	best_features = line.split()
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

	list_participant = os.listdir(path)
	list_participant.sort()

	participant_label = []
	num_sequence = []
	testing = 1
	save = 0
	ratio = [70, 30, 0]
	nbr_cross_val = 10
	test_generalisation = 0
	test_iteration = 0

	nbr_features = 50
	

	list_participant = ['541', '909', '3327', '5124', '5521', '5535', '8410', '9266', '9875']

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


			list_features = list(data_base.columns.values)
			del list_features[0:2]
			dim_features = np.ones(len(list_features))

			time = data_base['timestamps']

			
			labels, states = ref_data.load_labels_refGT(time, name_track, 'labels_3A')

			real_labels.append(labels)
			data_win2.append(data_base[list_features].as_matrix())
			timestamps.append(time)

			

			for state in states:
				if(state not in list_states):
					list_states.append(state)
					list_states = sorted(list_states)
			i += 1

	####################################################################################
	# Feature Selection
	####################################################################################
	obs = []
	obs = data_win2[0]
	lengths = []
	lengths.append(len(data_win2[0]))
	labels = real_labels[0]

	for i in range(1, len(data_win2)):
		obs = np.concatenate([obs, data_win2[i]])
		lengths.append(len(data_win2[i]))
		labels = np.concatenate([labels, real_labels[i]])


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

	del sorted_features_fisher[50:]

	score_totaux.to_csv('fisher_' + name_track + ".csv", index=False)
	best_features = sorted_features_fisher[0:nbr_features]

	file_name = 'wrapper_' + name_track + ".csv"
	wrapper_df = pd.read_csv(file_name)

	sorted_features_wrapper = []

	for n_features in range(1, 21):
		for i in range(len(wrapper_df)):
			line = wrapper_df['best_features'].values[i]
			line = re.sub(',', '', line)
			line = line.replace("[","")
			line = line.replace("]","")
			line = line.replace("'","")
			best_features = line.split()
			if(len(best_features) == n_features):
				sorted_features_wrapper.append(best_features)
				break

	score_F1_fisher = []
	sd_F1_fisher = []
	list_features_fisher = []

	score_F1_wrapper= []
	sd_F1_wrapper= []
	list_features_wrapper = []

	index_nbr_f = []

	score_wrapper = np.zeros((len(sorted_features_wrapper), nbr_cross_val))
	score_fisher = np.zeros((len(sorted_features_wrapper), nbr_cross_val))

	score_total = []
	list_method = []

	for nbr_features in range(1, len(sorted_features_wrapper)+1):
		if(test_iteration == 0):
			best_features_wrapper = find_best_features('wrapper_dimensions_' + name_track + ".csv")
		else:
			
			best_features_wrapper = sorted_features_wrapper[nbr_features-1]
			

		best_features_fisher = sorted_features_fisher[0:nbr_features]

		data_win = deepcopy(data_win2)

		dim_features = np.ones(nbr_features)

		confusion_matrix_fisher = np.zeros((len(list_states), len(list_states)))
		confusion_matrix_wrapper = np.zeros((len(list_states), len(list_states)))

		TP = 0
		
		short = 0

		transition_error = []
		short_transition_error = 0

		MCC = 0
		F1_fisher = []
		F1_wrapper = []

		F1_f = []
		F1_w = []

		if(nbr_cross_val == 0):
			model = ModelHMM()
			model.train(data_win, real_labels, best_features, dim_features)

			if(save):
				model.save_model(path_model, name_model, "load_handling")


		for n_subject in range(len(list_participant)):
			data_reduce = deepcopy(data_win)
			labels_reduce = deepcopy(real_labels)

			if(test_generalisation):
				data_gen = []
				labels_gen = []
				seq_subject = 0
				count = []
				for i in range(len(info_participant)):
					if(info_participant[i] == list_participant[n_subject]):
						data_gen.append(data_win[i])
						labels_gen.append(real_labels[i])

						count.append(i)

				del data_reduce[count[0]:count[-1]+1]
				del labels_reduce[count[0]:count[-1]+1]

			else:
				n_subject = len(list_participant)

			F1_wrapper_temp = 0
			F1_fisher_temp = 0


			for nbr_test in range(nbr_cross_val):
				total = 0
				data_ref, labels_ref, data_test, labels_test, id_train, id_test = tools.split_data_base2(data_reduce, labels_reduce, ratio)

				data_fisher = []
				data_wrapper = []

				data_win = deepcopy(data_win2)

				for id_subject in id_train:
					df = pd.DataFrame(data_win2[id_subject])
					df.columns = list_features

					data_fisher.append(df[best_features_fisher].values)
					data_wrapper.append(df[best_features_wrapper].values)

				model_fisher = ModelHMM()
				model_fisher.train(data_fisher, labels_ref, best_features_fisher, dim_features)

				model_wrapper = ModelHMM()
				model_wrapper.train(data_wrapper, labels_ref, best_features_wrapper, dim_features)


				if(save):
					model.save_model(path_model, name_model, "load_handling")


				#### Test
				data_fisher = []
				data_wrapper = []
				for id_subject in id_test:
					df = pd.DataFrame(data_win2[id_subject])
					df.columns = list_features

					data_fisher.append(df[best_features_fisher].values)
					data_wrapper.append(df[best_features_wrapper].values)

				predict_labels_fisher, proba = model_fisher.test_model(data_fisher)
				predict_labels_wrapper, proba = model_wrapper.test_model(data_wrapper)

				time_test = []
				for id_subject in id_test:
					time_test.append(timestamps[id_subject])

				for i in range(len(labels_test)):
					conf_mat = tools.compute_confusion_matrix(predict_labels_fisher[i], labels_test[i], list_states)
					confusion_matrix_fisher += conf_mat

					conf_mat = tools.compute_confusion_matrix(predict_labels_wrapper[i], labels_test[i], list_states)
					confusion_matrix_wrapper += conf_mat

					F1_fisher.append(tools.compute_F1_score(labels_test[i], predict_labels_fisher[i], list_states))
					F1_wrapper.append(tools.compute_F1_score(labels_test[i], predict_labels_wrapper[i], list_states))
					total += 1

				F1_f.append(np.mean(F1_fisher))
				F1_w.append(np.mean(F1_wrapper))

				F1_fisher_temp += np.mean(F1_fisher)
				F1_wrapper_temp += np.mean(F1_wrapper)
				

				index_nbr_f.append(nbr_features-1)
				index_nbr_f.append(nbr_features-1)

				score_total.append(np.mean(F1_fisher))
				list_method.append('Filter')
				score_total.append(np.mean(F1_wrapper))
				list_method.append('Wrapper')

			score_F1_fisher.append(F1_fisher_temp/nbr_cross_val)
			score_F1_wrapper.append(F1_wrapper_temp/nbr_cross_val)

			if(test_generalisation):
				predict_gen, proba = model.test_model(data_gen)
			
				for i in range(len(predict_gen)):
					conf_mat = tools.compute_confusion_matrix(predict_gen[i], labels_gen[i], list_states)
					confusion_matrix2 += conf_mat
			else:
				break
			


		if(testing):

			prec_total, recall_total, F1_score_fisher = tools.compute_score(confusion_matrix_fisher)
			prec_total, recall_total, F1_score_wrapper = tools.compute_score(confusion_matrix_wrapper)			# acc = tools.get_accuracy(confusion_matrix)

	

			if(test_generalisation):
				prec_total, recall_total, F1_score = tools.compute_score(confusion_matrix2)

			list_features_fisher.append(best_features_fisher)


			list_features_wrapper.append(best_features_wrapper)

			print(score_F1_fisher, F1_score_fisher)
			print(list_features_fisher)

			if(test_iteration):

				score_totaux_fisher = pd.DataFrame(
				{'nbr_features': np.arange(1,nbr_features+1),
				 'features': list_features_fisher,
				 'score': score_F1_fisher,
				})


				score_totaux_wrapper = pd.DataFrame(
				{'nbr_features': np.arange(1,nbr_features+1),
				 'features': list_features_wrapper,
				 'score': score_F1_wrapper,
				})	

				score_totaux_fisher.to_csv('fisher' + '_dimensions' + '_' + name_track + "2.csv", index=False)
				score_totaux_wrapper.to_csv('wrapper' + '_dimensions' + '_' + name_track + "2.csv", index=False)

				print('Fisher', F1_f, F1_score_fisher)
				print('Wrapper', F1_w, F1_score_wrapper)

				if(test_iteration):
					score_totaux = pd.DataFrame(
					{'features': index_nbr_f,
					 'method' : list_method,
					 'score': score_total,
					})

					score_totaux.to_csv('score' + '_dimensions' + '_' + name_track + "2.csv", index=False)
				
			else:
				tools.plot_confusion_matrix2('', name_track, confusion_matrix_wrapper, list_states, save = 1, all_in_one = 1)
				tools.plot_confusion_matrix2('', name_track, confusion_matrix_fisher, list_states, save = save, all_in_one = 1)
				df_conf = pd.DataFrame(confusion_matrix_wrapper, index = list_states, columns = list_states)
				df_conf.to_csv('confusion_' + method + '_' + name_track + ".csv", index=True)
				break

			

	plt.show()
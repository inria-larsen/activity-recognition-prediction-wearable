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

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
	# Prepare the data base
	name_track = 'detailed_posture'
	# name_track = 'general_posture'

	yarp.Network.init()
	rf = yarp.ResourceFinder()
	rf.setDefaultContext("online_recognition")
	rf.setDefaultConfigFile("selection_features.ini")
	rf.configure(sys.argv)

	path = rf.find('path_data_base').toString()
	path_model = rf.find('path_model').toString()
	name_model = rf.find('name_model').toString()

	# list_participant = ['909', '5521', '541', '3327']
	participant_label = []
	num_sequence = []
	list_participant = ['909']
	testing = 1
	save = 0
	ratio = [70, 30, 0]
	nbr_cross_val = 10

	# list_participant = ['5521']


	# # path_video = '/home/amalaise/Documents/These/experiments/AnDy-LoadHandling/annotation/Videos_Xsens/Participant_541/Participant_541_Setup_A_Seq_3_Trial_1.mp4'
	# path_video = '/home/amalaise/Documents/These/experiments/AnDy-LoadHandling/annotation/Videos_Xsens/Participant_909/Participant_909_Setup_A_Seq_1_Trial_1.mp4'


	# id_test = 0

	data_base = []

	print('Loading data...')

	for participant, nbr in zip(list_participant, range(len(list_participant))):
		print('Loading: ' + participant)
		data_base.append(DataBase(path + '/' + participant))
		data_base[nbr].load_mvnx_data()


		signals = rf.findGroup("Signals").tail().toString().replace(')', '').replace('(', '').split(' ')
		nb_port = int(len(signals))
		nb_active_port = 0

		list_features, dim_features = data_base[nbr].add_signals_to_dataBase(rf)

		if('eglove' in signals):
			info_signal = rf.findGroup('eglove')
			list_items = info_signal.findGroup('list').tail().toString().split(' ')
			glove_on = int(info_signal.find('enable').toString())
			if(glove_on):
				data_glove, time_glove = data_base[nbr].add_data_glove(info_signal)


	print('Data Loaded')


	# ###################################################
			
	timestamps = []
	data_win = []
	data_glove2 = []
	timestamps_glove = []
	real_labels = []
	real_labels_detailed = []
	list_states = []
	list_states_detailed = []

	window_size = float(rf.find('slidding_window_size').toString())

	for j in range(len(data_base)):
		sub_data = pr.concatenate_data(data_base[j], list_features)
		n_seq = data_base[j].get_nbr_sequence()
		t = []

		for i in range(n_seq):
			t.append(data_base[j].get_data_by_features('time', i))
			t_mocap = t[i].tolist()
			mocap_data = sub_data[i].tolist()

			if(glove_on):
				t_glove = time_glove[i]
				glove_data = data_glove[i].tolist()

				t_ = 0
				while(t_glove[t_] < t_mocap[0]):
					t_ += 1
				t_start = t_


				while(t_mocap[t_] < t_glove[-1]):
					t_ += 1
				t_end = t_+1

				del glove_data[0:t_start]
				del t_glove[0:t_start]

				del mocap_data[t_end:]
				del t_mocap[t_end:]

				data_force = np.zeros((len(t_mocap), np.shape(glove_data)[1]))

				count = 0

				for k in range(len(t_mocap)):
					data_force[k] = glove_data[count]
					if(t_glove[count] < t_mocap[k]):
						count += 1
						if(count == len(glove_data)):
							break

				data_out, timestamps_out = pr.slidding_window(data_force, t_mocap, window_size)
				data_glove[i] = data_out


			data_out, timestamps_out = pr.slidding_window(mocap_data, t_mocap, window_size)
			t[i] = timestamps_out
			data_win.append(data_out)
			participant_label.append(j)
			num_sequence.append(i + 1)
			timestamps.append(timestamps_out)

			if(glove_on):
				data_win[i] = np.concatenate((data_win[i], data_glove[i]) , axis = 1)

	
		data_base[j].load_labels_ref(name_track)
		labels = data_base[j].get_real_labels(t)


		for seq_labels in labels:
			real_labels.append(seq_labels)

		states = data_base[j].get_list_states()

		for state in states:
			if(state not in list_states):
				list_states.append(state)
				list_states = sorted(list_states)



		# data_base[j].load_labels_ref('detailed_posture')
		# labels_detailed = data_base[j].get_real_labels(t)

		# for seq_labels_detailed in labels_detailed:
		# 	real_labels_detailed.append(seq_labels_detailed)

		# states = data_base[j].get_list_states()

		# for state in states:
		# 	if(state not in list_states_detailed):
		# 		list_states_detailed.append(state)
		# 		list_states_detailed = sorted(list_states_detailed)

	if(glove_on):
		list_features.append('gloveForces')
		dim_features.append(4)
	
	F1_S = 0


	print(np.shape(data_win))

	obs = []
	obs = data_win[0]
	lengths = []
	lengths.append(len(data_win[0]))
	labels = real_labels[0]

	for i in range(1, len(data_win)):
		obs = np.concatenate([obs, data_win[i]])
		lengths.append(len(data_win[i]))
		labels = np.concatenate([labels, real_labels[i]])


	list_states, labels = np.unique(labels, return_inverse=True)
	n_states = len(list_states)

	Y = labels.reshape(-1, 1) == np.arange(len(list_states))
	end = np.cumsum(lengths)
	start = end - lengths

	# Compute the initial probabilities
	init_prob = Y[start].sum(axis=0)/Y[start].sum()
	# init_prob = np.ones(self.n_states)/self.n_states

	n_feature = sum(dim_features)

	# Compute the emission distribution
	# Mu, covars = tools.mean_and_cov(obs, labels, n_states, n_feature)

	data = [[]]
	for i in range(n_states - 1):
		data.append([])

	for i in range(len(labels)):
		num_state = labels[i]
		if(len(data[num_state])<=0):
			data[num_state].append(obs[i])
		else:
			data[num_state] = np.vstack((data[num_state], obs[i]))





	####################################################################################
	# Feature Selection
	####################################################################################

	list_all_features = []

	for feature, dim in zip(list_features, dim_features):
		if(dim > 1):
			for i in range(dim):
				list_all_features.append(feature + '_' + str(i))
		else:
			list_all_features.append(feature)

	print(np.shape(obs))

	df = pd.DataFrame(obs)
	df.columns = list_all_features
	df.to_csv('save_features' + ".csv", index=False)


	df2 = pd.DataFrame(labels)
	df2.columns = ['state']
	df2.to_csv('states' + ".csv", index=False)

	f_score = tools.fisher_score2(data, obs)
	list_id_sort = np.argsort(f_score)

	for id_sort in list_id_sort:
		print(list_all_features[id_sort], f_score[id_sort])








	# for nbr_test in range(nbr_cross_val):
	# 	confusion_matrix = np.zeros((len(list_states), len(list_states)))

	# 	# data_ref, labels_ref, data_test, labels_test, data_val, labels_val, id_train, id_test, id_val = tools.split_data_base(data_win, real_labels, ratio)
	# 	data_ref, labels_ref, data_test, labels_test, id_train, id_test = tools.split_data_base2(data_win, real_labels, ratio)





	# 			model = ModelHMM()
	# 			model.train(data_ref, labels_ref, sub_list_features, sub_dim_features)

	# 			if(save):
	# 				model.save_model(path_model, name_model, "load_handling")


	# 			#### Test

	# 			# ref_labels_detailed = []

	# 			time_test = []
	# 			for id_subject in id_test:
	# 				time_test.append(timestamps[id_subject])
	# 				# ref_labels_detailed.append(real_labels_detailed[id_subject])


	# 			predict_labels, proba = model.test_model(data_test)

	# 			for i in range(len(predict_labels)):
	# 				conf_mat = tools.compute_confusion_matrix(predict_labels[i], labels_test[i], list_states)
	# 				confusion_matrix += conf_mat

	# 			prec_total, recall_total, F1_score = tools.compute_score(confusion_matrix)
	# 			acc = tools.get_accuracy(confusion_matrix)
	# 			# print(confusion_matrix)

	# 			F1_S += F1_score/nbr_cross_val



	# 		if(len(score) == 0):
	# 			score.append(F1_S)
	# 			best_features.append(sub_list_features)
	# 			dim_best.append(sub_dim_features)
	# 		else:
	# 			for num in range(len(score)):
	# 				if(F1_S > score[num]):
	# 					score.insert(num, F1_S)
	# 					best_features.insert(num, sub_list_features)
	# 					dim_best.insert(num, sub_dim_features)
	# 					break

	# 				if(num == len(score)-1):
	# 					score.append(F1_S)
	# 					best_features.append(sub_list_features)
	# 					dim_best.append(sub_dim_features)


	# score_totaux = pd.DataFrame(
	# 	{'best_features': best_features,
	# 	 'score': score,
	# 	})
	# score_totaux.to_csv('results' + ".csv", index=False)

	plt.show()



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
	# name_track = 'current_action'
	# name_track = 'words'

	name_track = 'detailed_posture'

	yarp.Network.init()
	rf = yarp.ResourceFinder()
	rf.setDefaultContext("online_recognition")
	rf.setDefaultConfigFile("default.ini")
	rf.configure(sys.argv)

	path = rf.find('path_data_base').toString()
	path_model = rf.find('path_model').toString()
	name_model = rf.find('name_model').toString()

	print(path)

	list_participant = ['909', '5521', '541', '3327']
	participant_label = []
	num_sequence = []
	# list_participant = ['909']
	testing = 1
	save = 0
	ratio = [70, 30, 0]
	nbr_cross_val = 1

	# list_participant = ['0']


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
			glove_on = int(info_signal.find('enable').toString())
			if(glove_on):
				data_glove, time_glove = data_base[nbr].add_data_glove(info_signal)


	print('Data Loaded')


###################################################
	
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

				print(t_glove[0], t_mocap[0])
				print(t_glove[-1], t_mocap[-1])

				t_ = 0
				while(t_glove[t_] < t_mocap[0]):
					t_ += 1
				t_start = t_


				while(t_mocap[t_] < t_glove[-1]):
					t_ += 1
					if(t_ == len(t_mocap)):
						break

				t_end = t_+1

				del glove_data[0:t_start]
				del t_glove[0:t_start]

				# del mocap_data[t_end:]
				# del t_mocap[t_end:]

				del glove_data[t_end:]
				del t_glove[t_end:]

				data_force = np.zeros((len(t_mocap), np.shape(glove_data)[1]))

				count = 0

				for k in range(len(t_mocap)):
					data_force[k] = glove_data[count]
					if(t_glove[count] < t_mocap[k]):
						count += 1
						if(count == len(glove_data)):
							break
				# data_out, timestamps_out = pr.slidding_window(glove_data, t_glove, window_size)

				data_out, timestamps_out = pr.slidding_window(data_force, t_mocap, window_size)
				data_glove[i] = data_out

			if(glove_on):
				data_win.append(data_glove[i])

			else:
				data_out, timestamps_out = pr.slidding_window(mocap_data, t_mocap, window_size)
				data_win.append(data_out)

			t[i] = timestamps_out
			participant_label.append(j)
			num_sequence.append(i + 1)
			timestamps.append(timestamps_out)

			

	
		data_base[j].load_labels_ref(name_track)
		labels = data_base[j].get_real_labels(t)


		for seq_labels, i  in zip(labels, range(len(labels))):
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
		list_features = ['gloveForces']
		# list_features.append('gloveAngles')
		dim_features = [4]
		# dim_features.append(3)

		# data_win = data_glove

		# list_features = ['gloveForces']
		# dim_features = [7]
		
	scores = np.zeros((nbr_cross_val, 4))
	results_prec = np.zeros((nbr_cross_val, len(list_states)))
	results_recall = np.zeros((nbr_cross_val, len(list_states)))
	results_F1 = np.zeros((nbr_cross_val, len(list_states)))

	TP = 0
	total = 0
	short = 0

	transition_error = []
	short_transition_error = 0



	for nbr_test in range(nbr_cross_val):
		confusion_matrix = np.zeros((len(list_states), len(list_states)))
		confusion_matrix2 = np.zeros((len(list_states), len(list_states)))

		# data_ref, labels_ref, data_test, labels_test, data_val, labels_val, id_train, id_test, id_val = tools.split_data_base(data_win, real_labels, ratio)
		data_ref, labels_ref, data_test, labels_test, id_train, id_test = tools.split_data_base2(data_win, real_labels, ratio)

		model = ModelHMM()
		model.train(data_ref, labels_ref, list_features, dim_features)

		if(save):
			model.save_model(path_model, name_model, "load_handling")


		#### Test

		# ref_labels_detailed = []

		time_test = []
		for id_subject in id_test:
			time_test.append(timestamps[id_subject])
			# ref_labels_detailed.append(real_labels_detailed[id_subject])


		predict_labels, proba = model.test_model(data_test)

		pred = [[]]
		g_truth = [[]]

		for k in range(len(predict_labels)):

			for t in range(len(predict_labels[k])):
				# print(k, t, len(predict_labels[k]), len(g_truth), len(list_states))
				g_truth[k].append(list_states.index(labels_test[k][t]))
				pred[k].append(list_states.index(predict_labels[k][t]))

			if(len(pred) < len(predict_labels)):
				pred.append([])
				g_truth.append([])

			plt.figure()
			plt.subplot(211)
			plt.plot(time_test[k], data_test[k])
			plt.subplot(212)
			plt.plot(time_test[k], pred[k], color = 'red')
			plt.plot(time_test[k], g_truth[k], color = 'green')
			y_axis = np.arange(0, len(list_states), 1)
			plt.yticks(y_axis, list_states)
			plt.title(list_participant[participant_label[id_test[k]]] + '_' + str(num_sequence[id_test[k]]))

		time, ground_truth, prediction, id_sample_start, id_sample_end = tools.prepare_segment_analysis(time_test, predict_labels, labels_test)


		predict_labels2 = deepcopy(predict_labels)



		for i in range(len(ground_truth)):
			length = len(ground_truth[i])
			for t in range(length):
				if(t < len(prediction[i])-1):
					if(time[i][t+1] - time[i][t] < 1.0):
						short += 1
						

				if(prediction[i][t] == ground_truth[i][t]):
					TP += time[i][t]

				else:
					if(t < len(prediction[i])-1 and t > 0):
						print(prediction[i][t], ground_truth[i][t], time[i][t+1] - time[i][t], id_sample_start[i][t], id_sample_end[i][t])

						if((prediction[i][t-1] == ground_truth[i][t-1] and prediction[i][t+1] == ground_truth[i][t+1]) and
							((prediction[i][t] == prediction[i][t-1] and ground_truth[i][t] == ground_truth[i][t+1]) or
							(prediction[i][t] == prediction[i][t+1] and ground_truth[i][t] == ground_truth[i][t-1]))):

							transition_error.append(time[i][t+1] - time[i][t])

							if(time[i][t+1] - time[i][t] < 0.55):
								# TP += 1
								short_transition_error += 1
								prediction[i][t] = ground_truth[i][t]

								predict_labels2[i][id_sample_start[i][t]:id_sample_end[i][t]+1] = labels_test[i][id_sample_start[i][t]:id_sample_end[i][t]+1]


					# else:
						# print('\n')
						# print(t)
						# print(prediction[i][t-1], prediction[i][t], prediction[i][t+1])
						# print(ground_truth[i][t-1], ground_truth[i][t], ground_truth[i][t+1])
						# print(time[i][t+1] - time[i][t])

				total += time[i][t]

			conf_mat = tools.compute_confusion_matrix(predict_labels2[i], labels_test[i], list_states)
			confusion_matrix += conf_mat

			conf_mat = tools.compute_confusion_matrix(predict_labels[i], labels_test[i], list_states)
			prec, recall, F1 = tools.compute_score(conf_mat)
			scores[nbr_test, 0:3] = tools.compute_score(conf_mat)
			prec_states, recall_states, F1_states = tools.compute_score_by_states(conf_mat)
			results_prec[nbr_test, :] = prec_states
			results_recall[nbr_test] = recall_states
			results_F1[nbr_test] = F1_states
			scores[nbr_test, 3]= TP/total
			confusion_matrix2 += conf_mat

	prec_total, recall_total, F1_score = tools.compute_score(confusion_matrix)
	acc = tools.get_accuracy(confusion_matrix)
	print(confusion_matrix)
	print(F1_score, recall_total, prec_total)

	# prec_total, recall_total, F1_score = tools.compute_score(confusion_matrix2)
	prec_total, recall_total, F1_score = tools.compute_score(confusion_matrix2)
	acc = tools.get_accuracy(confusion_matrix2)
	print(confusion_matrix2)
	print(F1_score, recall_total, prec_total)

	print(TP/total, short_transition_error/total)


	df_prec = pd.DataFrame(results_prec.T)
	df_recall = pd.DataFrame(results_recall.T)
	df_F1 = pd.DataFrame(results_F1.T)
	# df.columns = ['id', 'com_z']
	plt.subplot(311)
	plt.boxplot(df_prec)
	plt.subplot(312)
	plt.boxplot(df_recall)
	plt.subplot(313)
	plt.boxplot(df_F1)


	print(len(transition_error))

					# print(TP/total)


		# print(list_features, best_features)
		# del dim_features[list_features.index(best_features[0])]
		# print('b', list_features)
		# del list_features[list_features.index(best_features[0])]
		
		

	tools.plot_confusion_matrix2('', '', confusion_matrix, list_states)

	tools.plot_confusion_matrix2('', '', confusion_matrix2, list_states)


#############################################################################


		# confusion_matrix = np.zeros((len(list_states_detailed), len(list_states_detailed)))
		# data_model_detailed = []

		# for sequence in (range(len(data_test))):
		# 	data_model_detailed.append(np.concatenate((data_test[sequence], proba[sequence]), axis = 1))
		# 	print('a', sequence, np.shape(data_model_detailed[sequence]), len(ref_labels_detailed[sequence]))

		# list_features.append('scores')
		# dim_features.append(3)

		
		# model_detailed = ModelHMM()
		# model_detailed.train(data_model_detailed, ref_labels_detailed, list_features, dim_features)


		# time_val = []
		# val_labels_detailed = []
		# for id_subject in id_val:
		# 	time_val.append(timestamps[id_subject])
		# 	val_labels_detailed.append(real_labels_detailed[id_subject])

		# predict_labels, proba = model.test_model(data_val)

		# data_val_detailed = []

		# for sequence in (range(len(data_val))):
		# 	data_val_detailed.append(np.concatenate((data_val[sequence], proba[sequence]), axis = 1))


		# predict_labels, proba = model_detailed.test_model(data_val_detailed)

		# print(list_states_detailed)

		# for k in range(len(predict_labels)):
		# 	g_truth = []
		# 	pred = []
		# 	for t in range(len(predict_labels[k])):
		# 		g_truth.append(list_states_detailed.index(val_labels_detailed[k][t]))
		# 		pred.append(list_states_detailed.index(predict_labels[k][t]))

		# 	plt.figure()
		# 	plt.plot(pred, color = 'red')
		# 	plt.plot(g_truth, color = 'green')
		# 	y_axis = np.arange(0, len(list_states_detailed), 1)
		# 	plt.yticks(y_axis, list_states_detailed)
		# 	plt.title(list_participant[participant_label[id_val[k]]] + '_' + str(num_sequence[id_val[k]]))

		# time, ground_truth, prediction = tools.prepare_segment_analysis(time_val, predict_labels, val_labels_detailed)

		# TP = 0
		# total = 0

		# for i in range(len(ground_truth)):
		# 	for t in range(len(ground_truth[i])):
		# 		if(prediction[i][t] == ground_truth[i][t]):
		# 			TP += 1

		# 		elif((prediction[i][t-1] == ground_truth[i][t-1] and prediction[i][t+1] == ground_truth[i][t+1]) and
		# 			((prediction[i][t] == prediction[i][t-1] and ground_truth[i][t] == ground_truth[i][t+1]) or
		# 			(prediction[i][t] == prediction[i][t+1] and ground_truth[i][t] == ground_truth[i][t-1]))):


		# 			if(time[i][t+1] - time[i][t] < 1.0):
		# 				TP += 1
		# 				prediction[i][t] = ground_truth[i][t]
		# 			# else:
		# 				# print('\n')
		# 				# print(t)
		# 				# print(prediction[i][t-1], prediction[i][t], prediction[i][t+1])
		# 				# print(ground_truth[i][t-1], ground_truth[i][t], ground_truth[i][t+1])
		# 				# print(time[i][t+1] - time[i][t])

		# 		total += 1

		# 	conf_mat = tools.compute_confusion_matrix(prediction[i], ground_truth[i], list_states_detailed)
		# 	confusion_matrix += conf_mat



		# prec_total, recall_total, F1_score = tools.compute_score(confusion_matrix)
		# acc = tools.get_accuracy(confusion_matrix)
		# print(confusion_matrix)
		# print(prec_total, recall_total, F1_score)
		# print(TP/total)

		# tools.plot_confusion_matrix2('', '', confusion_matrix, list_states_detailed)
























		# model.save_model(path_model, name_model, "load_handling")

		# color = ['red', 'blue', 'yellow', 'green']

		

		# for state in list_states:
		# 	test_labels = []
		# 	com_value = []

		# 	for i in range(len(data_test)):
		# 		obs = data_test[i]
		# 		labels = labels_test[i]

		# 		for t in range(len(obs)):
		# 			if(labels[t] == state):
		# 				test_labels.append([list_participant[participant_label[i]], obs[t][0]])


		# 	tab_ = np.asarray(test_labels)

		# 	df = pd.DataFrame(tab_)
		# 	df.columns = ['id', 'com_z']
		# 	df.to_csv(state + ".csv", index=False)



			# plt.subplot(212)
			# plt.plot(results)
			# y_axis = np.arange(0, len(list_states), 1)
			# plt.yticks(y_axis, list_states)


			
			# real_index = []
			# for j in range(len(real_labels)):
			# 	real_index.append(list_states.index(real_labels[j]))

			# plt.plot(real_index)


		# pred_labels = []
		# for j in range(len(results)):
		# 	pred_labels.append(list_states[results[j]])

		# print((tg[0][-1] - tg[0][0])/len(tg[0]))


		# v_tools.video_sequence(real_labels, pred_labels, path_video, 'test.mp4')

	plt.show()



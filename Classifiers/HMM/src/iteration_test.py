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


def find_best_features(file_name):
	df = pd.read_csv(file_name)
	line = df['features'].values[df['score'].idxmax()] # Find the set of features with the best score
	print(line)
	line = re.sub(',', '', line)
	line = line.replace("[","")
	line = line.replace("]","")
	line = line.replace("'","")
	best_features = line.split()
	return best_features

def get_list_states(file_name):
	data, labels, time, list_states, list_features = tools.load_data(path + '/' + participant + '/', name_seq, 'detailed_posture', labels_folder + '_details')
	labels_details.append(labels)

	for state in list_states:
		if(state not in list_states_details):
			list_states_details.append(state)
			list_states_details = sorted(list_states_details)
	i += 1

	return




if __name__ == '__main__':
	# Prepare the data base
	yarp.Network.init()
	rf = yarp.ResourceFinder()
	rf.setDefaultContext("online_recognition")
	rf.setDefaultConfigFile("default.ini")
	rf.configure(sys.argv)

	method = 'wrapper'

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
	save = 1
	ratio = [70, 30, 0]
	nbr_cross_val = 10
	test_generalisation = 0

	list_participant = ['541', '909', '3327', '5124', '5521', '5535', '8410', '9266', '9875']
	# list_participant = ['541']



	# id_test = 0

	print('Loading data...')

	timestamps = []
	data_win = []
	labels_posture = []
	list_states_posture = []

	labels_details = []
	list_states_details = []

	labels_posture_detailed = []
	list_states_posture_detailed = []

	labels_current_action = []
	list_states_current_action = []

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

			data, labels, time, list_states, list_features = tools.load_data(path + '/' + participant + '/', name_seq, name_track, labels_folder)
			labels_posture.append(labels)
			data_win.append(data)
			timestamps.append(time)

			for state in list_states:
				if(state not in list_states_posture):
					list_states_posture.append(state)
					list_states_posture = sorted(list_states_posture)


			data, labels, time, list_states, list_features = tools.load_data(path + '/' + participant + '/', name_seq, 'detailed_posture', labels_folder + '_details')
			labels_details.append(labels)

			for state in list_states:
				if(state not in list_states_details):
					list_states_details.append(state)
					list_states_details = sorted(list_states_details)


			a, labels, b , list_states, c  = tools.load_data(path + '/' + participant + '/', name_seq, 'detailed_posture', labels_folder)
			labels_posture_detailed.append(labels)

			for state in list_states:
				if(state not in list_states_posture_detailed):
					list_states_posture_detailed.append(state)
					list_states_posture_detailed = sorted(list_states_posture_detailed)


			a, labels, b , list_states, c  = tools.load_data(path + '/' + participant + '/', name_seq, 'current_action', labels_folder)
			labels_current_action.append(labels)

			for state in list_states:
				if(state not in list_states_current_action):
					list_states_current_action.append(state)
					list_states_current_action = sorted(list_states_current_action)

	####################################################################################
	# Feature Selection
	####################################################################################

	if(method == 'fisher'):
		file_name = 'fisher_dimensions_'

	elif(method == 'wrapper'):
		file_name = 'wrapper_dimensions_'
		
	best_features_posture = find_best_features(file_name + 'general_posture.csv')
	best_features_details = find_best_features(file_name + 'details.csv')
	best_features_detailed_posture = find_best_features(file_name + 'detailed_posture.csv')
	best_features_current_action = find_best_features(file_name + 'current_action.csv')


	print(best_features_posture, best_features_details, best_features_detailed_posture)

	####################################################################################
	# Model
	####################################################################################

	scores = np.zeros((nbr_cross_val, 4))
	results_prec = np.zeros((nbr_cross_val, len(list_states_posture)))
	results_recall = np.zeros((nbr_cross_val, len(list_states_posture)))
	results_F1 = np.zeros((nbr_cross_val, len(list_states_posture)))

	data_posture = deepcopy(data_win)
	data_details = deepcopy(data_win)
	data_detailed_posture = deepcopy(data_win)
	data_current_action = deepcopy(data_win)

	for i in range(len(data_win)):
		df = pd.DataFrame(data_win[i])
		df.columns = list_features

		data_posture[i] = df[best_features_posture].values
		data_details[i] = df[best_features_details].values
		data_detailed_posture[i] = df[best_features_detailed_posture].values
		data_current_action[i] = df[best_features_current_action].values


	MCC_combined = []
	MCC_detailed = []
	MCC_general = []
	MCC_details = []

	F1_combined = []
	F1_detailed = []
	F1_general = []
	F1_details = []
	F1_action = []
	F1 = 0
	total = 0

	for n_subject in range(len(list_participant)):
		data_reduce_posture = deepcopy(data_posture)
		labels_reduce_posture = deepcopy(labels_posture)
		data_reduce_detailted = deepcopy(data_detailed_posture)
		labels_reduce_detailed = deepcopy(labels_posture_detailed)
		data_reduce_details = deepcopy(data_details)
		labels_reduce_details = deepcopy(labels_details)
		data_reduce_action = deepcopy(data_current_action)
		labels_reduce_action = deepcopy(labels_current_action)

		if(test_generalisation):
			data_gen_posture = []
			labels_gen_posture = []
			data_gen_detailed = []
			labels_gen_detailed = []
			data_gen_details = []
			labels_gen_details = []
			data_gen_action = []
			labels_gen_action= []
			count = []
			for i in range(len(info_participant)):
				if(info_participant[i] == list_participant[n_subject]):
					data_gen_posture.append(data_posture[i])
					labels_gen_posture.append(labels_posture[i])
					data_gen_detailed.append(data_detailed_posture[i])
					labels_gen_detailed.append(labels_posture_detailed[i])
					data_gen_details.append(data_details[i])
					labels_gen_details.append(labels_details[i])
					data_gen_action.append(data_current_action[i])
					labels_gen_action.append(labels_current_action[i])
					count.append(i)

			del data_reduce_posture[count[0]:count[-1]+1]
			del labels_reduce_posture[count[0]:count[-1]+1]
			del data_reduce_detailted[count[0]:count[-1]+1]
			del labels_reduce_detailed[count[0]:count[-1]+1]
			del data_reduce_details[count[0]:count[-1]+1]
			del labels_reduce_details[count[0]:count[-1]+1]
			del data_reduce_action[count[0]:count[-1]+1]
			del labels_reduce_action[count[0]:count[-1]+1]

		else:
			n_subject = len(list_participant)

		for nbr_test in range(nbr_cross_val):
			data_ref, labels_ref, data_test, labels_test, id_train, id_test = tools.split_data_base2(data_reduce_posture, labels_reduce_posture, ratio)
			
			model = ModelHMM()
			model.train(data_ref, labels_ref, best_features_posture, np.ones(len(best_features_posture)))

			labels_ref_details = []
			data_details_ref = []
			labels_ref_detailed_posture = []
			data_ref_detailed_posture = []
			labels_ref_action= []
			data_ref_action = []

			if(test_generalisation):
				for id_subject in id_train:
					labels_ref_details.append(labels_reduce_details[id_subject])
					data_details_ref.append(data_reduce_details[id_subject])
					labels_ref_detailed_posture.append(labels_reduce_detailed[id_subject])
					data_ref_detailed_posture.append(data_reduce_detailted[id_subject])
					labels_ref_action.append(labels_reduce_action[id_subject])
					data_ref_action.append(data_reduce_action[id_subject])

			else:
				for id_subject in id_train:
					labels_ref_details.append(labels_details[id_subject])
					data_details_ref.append(data_details[id_subject])
					labels_ref_detailed_posture.append(labels_posture_detailed[id_subject])
					data_ref_detailed_posture.append(data_detailed_posture[id_subject])
					labels_ref_action.append(labels_reduce_action[id_subject])
					data_ref_action.append(data_reduce_action[id_subject])


				

			model_details = ModelHMM()
			model_details.train(data_details_ref, labels_ref_details, best_features_details, np.ones(len(best_features_details)))

			model_detailed_posture = ModelHMM()
			model_detailed_posture.train(data_ref_detailed_posture, labels_ref_detailed_posture, best_features_detailed_posture, np.ones(len(best_features_detailed_posture)))
			
			model_action = ModelHMM()
			model_action.train(data_ref_action, labels_ref_action, best_features_current_action, np.ones(len(best_features_current_action)))

			if(test_generalisation):
				break

			else:
				time_test = []
				labels_test_details = []
				data_details_test = []
				labels_test_detailed_posture = []
				data_test_detailed_posture = []
				labels_test_action = []
				data_test_action = []
				accu = 0
				total_ = 0
				for id_subject in id_test:
					time_test.append(timestamps[id_subject])
					labels_test_details.append(labels_details[id_subject])
					data_details_test.append(data_details[id_subject])
					labels_test_detailed_posture.append(labels_posture_detailed[id_subject])
					data_test_detailed_posture.append(data_detailed_posture[id_subject])
					labels_test_action.append(labels_current_action[id_subject])
					data_test_action.append(data_current_action[id_subject])


				predict_labels, proba = model.test_model(data_test)
				predict_labels_details, proba_details = model_details.test_model(data_details_test)
				predict_labels_detailed_posture, proba = model_detailed_posture.test_model(data_test_detailed_posture)
				predict_labels_action, proba = model_action.test_model(data_test_action)

			# time, ground_truth, prediction, id_sample_start, id_sample_end = tools.prepare_segment_analysis(time_test, predict_labels, labels_test, id_test)
			# predict_labels2 = deepcopy(predict_labels)

				labels_final = deepcopy(predict_labels)

				list_states_posture_final= []

				for i in range(len(predict_labels)):
					length = len(predict_labels[i])
					for t in range(length):
						labels_final[i][t] += '_' + predict_labels_details[i][t]
						if(labels_final[i][t] == labels_test_detailed_posture[i][t]):
							accu += 1
						total_ += 1

					list_states, labels = np.unique(labels_final[i], return_inverse=True)
					for state in list_states:
						if(state not in list_states_posture_final):
							list_states_posture_final.append(state)
							list_states_posture_final = sorted(list_states_posture_final)

				for i in range(len(predict_labels)):
					MCC_combined.append(tools.compute_MCC_score(labels_test_detailed_posture[i], labels_final[i], list_states_posture_final))
					MCC_detailed.append(tools.compute_MCC_score(labels_test_detailed_posture[i], predict_labels_detailed_posture[i], list_states_posture_final))	
					MCC_general.append(tools.compute_F1_score(labels_test[i], predict_labels[i], list_states_posture))
					MCC_details.append(tools.compute_F1_score(labels_test_details[i], predict_labels_details[i], list_states_details))	
						
					F1_combined.append(tools.compute_F1_score(labels_test_detailed_posture[i], labels_final[i], list_states_posture_final))
					F1_detailed.append(tools.compute_F1_score(labels_test_detailed_posture[i], predict_labels_detailed_posture[i], list_states_posture_final))	
					F1_general.append(tools.compute_F1_score(labels_test[i], predict_labels[i], list_states_posture))
					F1_details.append(tools.compute_F1_score(labels_test_details[i], predict_labels_details[i], list_states_details))	
					F1_action.append(tools.compute_F1_score(labels_test_action[i], predict_labels_action[i], list_states_current_action))

					total += 1

				confusion_matrix = np.zeros((len(list_states_posture), len(list_states_posture)))
				confusion_matrix_details = np.zeros((len(list_states_posture_final), len(list_states_posture_final)))
				confusion_matrix_details2 = np.zeros((len(list_states_details), len(list_states_details)))
				confusion_matrix_detailed_posture = np.zeros((len(list_states_posture_detailed), len(list_states_posture_detailed)))
				confusion_matrix_action = np.zeros((len(list_states_current_action), len(list_states_current_action)))

				for i in range(len(predict_labels)):
					conf_mat = tools.compute_confusion_matrix(predict_labels[i], labels_test[i], list_states_posture)
					confusion_matrix += conf_mat

					conf_mat = tools.compute_confusion_matrix(labels_final[i], labels_test_detailed_posture[i], list_states_posture_final)
					confusion_matrix_details += conf_mat

					conf_mat = tools.compute_confusion_matrix(predict_labels_details[i], labels_test_details[i], list_states_details)
					confusion_matrix_details2 += conf_mat

					conf_mat = tools.compute_confusion_matrix(predict_labels_detailed_posture[i], labels_test_detailed_posture[i], list_states_posture_detailed)
					confusion_matrix_detailed_posture += conf_mat

					conf_mat = tools.compute_confusion_matrix(predict_labels_action[i], labels_test_action[i], list_states_current_action)
					confusion_matrix_action += conf_mat


				# df_details = pd.DataFrame(confusion_matrix_details, index = list_states_posture_detailed, columns = list_states_posture_detailed)
				# df_details.to_csv('confusion_' + 'details' + ".csv", index=True)

				prec_states, recall_states, F1_states = tools.compute_score_by_states(confusion_matrix)
				results_prec[nbr_test, :] = prec_states
				results_recall[nbr_test] = recall_states
				results_F1[nbr_test] = F1_states
				# scores[nbr_test, 3]= TP/total

		if(test_generalisation):
			predict_labels, proba = model.test_model(data_gen_posture)
			predict_labels_details, proba_details = model_details.test_model(data_gen_details)
			predict_labels_detailed_posture, proba = model_detailed_posture.test_model(data_gen_detailed)
			predict_labels_action, proba = model_detailed_posture.test_model(data_gen_action)

			labels_final = deepcopy(predict_labels)
			list_states_posture_final= []

			for i in range(len(predict_labels)):
				length = len(predict_labels[i])
				for t in range(length):
					labels_final[i][t] += '_' + predict_labels_details[i][t]
					# if(labels_final[i][t] == labels_test_detailed_posture[i][t]):
					# 	accu += 1
					# total_ += 1

				list_states, labels = np.unique(labels_final[i], return_inverse=True)
				for state in list_states:
					if(state not in list_states_posture_final):
						list_states_posture_final.append(state)
						list_states_posture_final = sorted(list_states_posture_final)

			for i in range(len(predict_labels)):
				F1_general.append(tools.compute_F1_score(labels_gen_posture[i], predict_labels[i], list_states_posture))
				F1_combined.append(tools.compute_F1_score(labels_gen_detailed[i], labels_final[i], list_states_posture_final))
				F1_detailed.append(tools.compute_F1_score(labels_gen_detailed[i], predict_labels_detailed_posture[i], list_states_posture_detailed))	
				F1_details.append(tools.compute_F1_score(labels_gen_details[i], predict_labels_details[i], list_states_details))
				F1_action.append(tools.compute_F1_score(labels_gen_action[i], predict_labels_action[i], list_states_current_action))	
					
				# print(np.shape(predict_gen[i]), np.shape(labels_gen[i]))
				# conf_mat = tools.compute_confusion_matrix(predict_gen[i], labels_gen_posture[i], list_states_posture)
				# confusion_matrix += conf_mat
			
		else:
			break
		# if(testing):

		# prec_total, recall_total, F1_score = tools.compute_score(confusion_matrix)
		# acc = tools.get_accuracy(confusion_matrix)

		# taux_rec = np.sum(np.diag(confusion_matrix))/np.sum(np.sum(confusion_matrix))
		# print(confusion_matrix)
		# print(F1_score, recall_total, prec_total, taux_rec)

		# prec_total, recall_total, F1_score = tools.compute_score(confusion_matrix_details)
		# taux_rec = np.sum(np.diag(confusion_matrix_details))/np.sum(np.sum(confusion_matrix_details))
		# acc = tools.get_accuracy(confusion_matrix_details)

	print('\ngeneral')
	# print('MCC', np.mean(MCC_general))
	print('F1-score', np.median(F1_general), np.mean(F1_general))
	# print('Acc', acc, taux_rec)

	print('\ndetails')
	# print('MCC', np.mean(MCC_details))
	print('F1-score', np.median(F1_details), np.mean(F1_details))
	# print('Acc', acc, taux_rec)

	print('\ncombined')
	# print('MCC', np.mean(MCC_combined))
	print('F1-score', np.median(F1_combined), np.mean(F1_combined))
	# print('Acc', acc, taux_rec)

	# prec_total, recall_total, F1_score = tools.compute_score(confusion_matrix_detailed_posture)
	# taux_rec = np.sum(np.diag(confusion_matrix_detailed_posture))/np.sum(np.sum(confusion_matrix_detailed_posture))
	# acc = tools.get_accuracy(confusion_matrix_detailed_posture)

	print('\ndetailed')
	# print('MCC', np.mean(MCC_detailed))
	print('F1-score', np.median(F1_detailed), np.mean(F1_detailed))
		# print('Acc', acc, taux_rec)

	print('\naction')
	# print('MCC', np.mean(MCC_detailed))
	print('F1-score', np.median(F1_action), np.mean(F1_action))

			



	# plt.figure()
	# plt.boxplot([F1_combined, F1_detailed, F1_general, F1_details])
	# plt.xlabel(['Combined posture', 'Detailed posture', 'General posture', 'Details'])


	# df_conf = pd.DataFrame(confusion_matrix_details, index = list_states_posture_final, columns = list_states_posture_final)
	# df_conf.to_csv('confusion_' + method + '_' + 'details_combined' + ".csv", index=True)

	# df_conf = pd.DataFrame(confusion_matrix_details2, index = list_states_details, columns = list_states_details)
	# df_conf.to_csv('confusion_' + method + '_' + 'details2' + ".csv", index=True)

	# df_conf = pd.DataFrame(confusion_matrix_detailed_posture, index = list_states_posture_detailed, columns = list_states_posture_detailed)
	# df_conf.to_csv('confusion_' + method + '_' + 'detailed_posture2' + ".csv", index=True)

	# df_conf = pd.DataFrame(confusion_matrix, index = list_states_posture, columns = list_states_posture)
	# df_conf.to_csv('confusion_' + method + '_' + 'general_posture2' + ".csv", index=True)

	df = pd.DataFrame([info_participant, F1_general, F1_details, F1_combined, F1_detailed]).T
	# df = pd.DataFrame([F1_general, F1_details, F1_combined, F1_detailed]).T
	df.columns = ['ID','F1_general', 'F1_details', 'F1_combined', 'F1_detailed']
	df.to_csv('score_' + method + '_comparison.csv', index=False)
	# print(df)


	tools.plot_confusion_matrix2('', 'current_action', confusion_matrix_action, list_states_current_action, save = 0, all_in_one=1)
	tools.plot_confusion_matrix2('', 'details_combined', confusion_matrix_details, list_states_posture_final, save = 0, all_in_one=1)	
	tools.plot_confusion_matrix2('', 'detailed_posture2', confusion_matrix_detailed_posture, list_states_posture_detailed, save = 0, all_in_one=1)
	tools.plot_confusion_matrix2('', 'general_posture2', confusion_matrix, list_states_posture, save = 0, all_in_one=1)	
	tools.plot_confusion_matrix2('', 'details2', confusion_matrix_details2, list_states_details, save = 0, all_in_one=1)




	path_video = '/home/amalaise/Documents/These/experiments/AnDy-LoadHandling/annotation/Videos_Xsens/Participant_' + str(info_participant[id_test[0]]) + '/' + info_sequences[id_test[0]] + '.mp4'

	# v_tools.video_sequence(labels_test_detailed_posture[0], labels_final[0], path_video, 'test.mp4')

	plt.show()



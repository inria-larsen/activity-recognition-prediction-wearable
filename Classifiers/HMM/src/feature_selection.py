from hmm_model import ModelHMM
from data_base import DataBase
import data_processing as pr
import numpy as np
import matplotlib.pyplot as plt
import sys
import visualization_tools as v_tools
import tools
import pandas as pd 
from copy import deepcopy
from sys import getsizeof
import os
import time
import pickle
import configparser
from scipy import stats
import argparse

import warnings
warnings.filterwarnings("ignore")

def ranking_features(df_score, list_tracks, method = 'mean'):
	if(len(list_tracks)==1):
		df_score = df_score.sort_values(by = list_tracks, ascending = False)

	else:
		if(method == 'gmean'):
			df_combine = pd.DataFrame({'score_global': stats.mstats.gmean(df_score.iloc[:,1:], axis=1)})
		elif(method == 'hmean'):
			df_combine = pd.DataFrame({'score_global': stats.hmean(df_score.iloc[:,1:], axis=1)})
		else:
			df_combine = pd.DataFrame({'score_global': df_score.mean(axis=1)})

		df_score = pd.concat([df_score, df_combine], axis=1)
		df_score = df_score.sort_values(by = ['score_global'], ascending = False)

	return df_score

if __name__ == '__main__':
	# arguments
	parser=argparse.ArgumentParser()
	parser.add_argument('--config', '-c', help='Configuration type', default="DEFAULT")
	args=parser.parse_args()
	config_type = args.config

	local_path = os.path.abspath(os.path.dirname(__file__))

	max_iter = 50

	config = configparser.ConfigParser()
	config.read('config_file/config_file_selection.ini')

	path = config["DEFAULT"]["path_data_base"]
	path = os.path.join(local_path, path)
	path_data_dump = config["DEFAULT"]["path_data_dump"]
	path_data_dump = os.path.join(local_path, path_data_dump)
	tracks = config[config_type]["list_tracks"].split(' ')
	path_save = config["DEFAULT"]["path_save"] + '/' + config_type + '/'
	file_name = config_type + '.csv'
	local_features_flag = config[config_type]["local_features"]
	method_sort = config[config_type]["method_sort"]

	all_tracks = ['general_posture', 'detailed_posture', 'details', 'current_action']

	if(not(os.path.isdir(path_save))):
		os.makedirs(path_save)


	path_data_root = path + '/xsens/allFeatures_csv/'

	list_participant = os.listdir(path_data_root)
	list_participant.sort()

	participant_label = []
	num_sequence = []
	testing = 1
	save = 0
	ratio = [70, 30, 0]
	nbr_cross_val = 3
	nbr_subsets_iter = 10

	print('Loading data...')

	timestamps = []

	real_labels = []
	list_states = []

	df_all_data = []

	# list_reduce_features = ['acceleration_RightLowerLeg_y', 'acceleration_T8_z', 'comPos_centerOfMass_z', 'jointAngle_jLeftT4Shoulder_y', 'orientation_RightHand_q1', 'orientation_RightShoulder_q1', 'position_L5_x', 'position_LeftForeArm_z', 'position_LeftHand_z', 'position_RightLowerLeg_z', 'position_RightUpperArm_z', 'velocityNorm']

	data_win2, real_labels, list_states, list_features = tools.load_data_from_dump(path_data_dump)

	if(local_features_flag == 'True'):
		list_reduce_features = tools.list_features_local(list_features)
	else:
		list_reduce_features = list_features

	id_track_rm = 0
	for num_track in range(len(all_tracks)):
		if(not(all_tracks[num_track] in tracks)):
			del real_labels[num_track - id_track_rm]
			del list_states[num_track - id_track_rm]
			id_track_rm += 1


	print(list_states)

	df_all_data = []

	for data, num_data in zip(data_win2, range(len(data_win2))):
		df_all_data.append(pd.DataFrame(data, columns = list_features))
		df_all_data[-1] = df_all_data[-1][list_reduce_features]
		data_win2[num_data] = df_all_data[-1][list_reduce_features].values

	list_features = list_reduce_features

	dim_features = np.ones(len(list_features))

	print('Data Loaded')

	flag = 0
	score = []
	best_features = []


	num_iteration = 1
	count_top_feature = 0
	while(os.path.isfile(path_save + '/' + file_name + str(num_iteration))):
		df_time = pd.read_csv(path_save + '/' + 'time_' +  file_name + str(num_iteration))
		if(num_iteration>1 and len(df_time)<nbr_subsets_iter):
			count_top_feature = len(df_time)
			break
		num_iteration += 1

	start = num_iteration - 1

	if(len(list_features)<max_iter):
		max_iter = len(list_features)

	for iteration in range(start, max_iter):
		save_time = []
		
		print('\n#############################')
		print('Iteration: ' + str(iteration+1))
		print('#############################')
		tic = time.clock()

		data_ref_all = [[]]
		labels_ref = [[[]]]
		data_test_all = [[]]
		labels_test = [[[]]]
		id_train = [[]]
		id_test = [[]]

		best_features_total = []
		score_total = [[]]
		

		for i in range(len(tracks)):
			if(i < len(tracks)-1):
				score_total.append([])

		for k in range(nbr_cross_val):

			for num_track in range(len(tracks)):
				if(num_track == 0):
					data_ref_all[k], labels_ref[num_track][k], data_test_all[k], labels_test[num_track][k], id_train[k], id_test[k] = tools.split_data_base2(data_win2, real_labels[num_track], ratio)
				else:
					for id_subject in id_train[k]:
						labels_ref[num_track][k].append(real_labels[num_track][id_subject])

					for id_subject in id_test[k]:
						labels_test[num_track][k].append(real_labels[num_track][id_subject])

				if(len(labels_ref) < len(tracks)):
					labels_ref.append([[]])
					labels_test.append([[]])

				if(k < nbr_cross_val-1):
					labels_ref[num_track].append([])
					labels_test[num_track].append([])

			if(k < nbr_cross_val-1):
				data_ref_all.append([])
				data_test_all.append([])
				id_train.append([])
				id_test.append([])

		if(iteration >= 1):
			top_list_feature = tools.get_best_features(path_save + '/' + file_name + str(iteration))[0:nbr_subsets_iter]

		else:
			top_list_feature = ['']

		skip_feature_num = 0
		for top_feature, num_feature in zip(top_list_feature, range(len(top_list_feature))):
			if(skip_feature_num < count_top_feature):
				if(skip_feature_num == 0):
					best_features_total = tools.get_best_features(path_save + '/' + file_name + str(iteration+1))
					score_df = pd.read_csv(path_save + '/' + file_name + str(iteration+1))
					for num_track in range(len(tracks)):
						score_total[num_track] = score_df[tracks[num_track]].values.tolist()
					save_time = pd.read_csv(path_save + '/' + 'time_' +  file_name + str(num_iteration)).values.tolist()
				skip_feature_num += 1
				continue
			count_top_feature = 0


			count = 0

			print('##')
			print(num_feature + 1, ' - ', top_feature)
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

					if(sub_list_features in best_features_total):
						continue

				else:
					sub_list_features = [feature]

		###################################################

				F1_S = [[]]
				for i in range(len(tracks)):
					if(i < len(tracks)-1):
						F1_S.append([])

				best_features_total.append(sub_list_features)
			
				for name_track, num_track in zip(tracks, range(len(tracks))):

					confusion_matrix = np.zeros((len(list_states[num_track]), len(list_states[num_track])))
					F1_score = []

					for nbr_test in range(nbr_cross_val):

						data_ref = []
						data_test = []

						for data in data_ref_all[nbr_test]:
							df_data = pd.DataFrame(data, columns = list_features)
							data_ref.append(df_data[sub_list_features].values)

						for data in data_test_all[nbr_test]:
							df_data = pd.DataFrame(data, columns = list_features)
							data_test.append(df_data[sub_list_features].values)

						model = ModelHMM()
						model.train(data_ref, labels_ref[num_track][nbr_test], sub_list_features, np.ones(len(sub_list_features)))

						#### Test
						predict_labels, proba = model.test_model(data_test)
				
						for i in range(len(predict_labels)):
							F1_score.append(tools.compute_F1_score(labels_test[num_track][nbr_test][i], predict_labels[i], list_states[num_track]))

					F1_S[num_track] = np.mean(F1_score)
					score_total[num_track].append(F1_S[num_track])

			score_totaux = pd.DataFrame(
				{'best_features': best_features_total})

			for name_track, num_track in zip(tracks, range(len(tracks))):
				df_track = pd.DataFrame(
					{ name_track: score_total[num_track]})
				score_totaux = pd.concat([score_totaux, df_track], axis=1)

			score_totaux = ranking_features(score_totaux, tracks, method_sort)

			score_totaux.to_csv(path_save + '/' + file_name + str(iteration+1), index=False)

			toc = time.clock()
			print('Time: ', toc - tic)
			save_time.append(toc -tic)
			time_totaux = pd.DataFrame({'time': save_time})
			time_totaux.to_csv(path_save + '/' + 'time_' +  file_name + str(iteration+1), index=False)

		score_total.append([])
		best_features_total.append([])
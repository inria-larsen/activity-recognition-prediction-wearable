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

	video_name = 'video2_' + name_track + '.mp4'

	list_participant = os.listdir(path)
	list_participant.sort()

	path_model = '/home/amalaise/Documents/These/code/activity-recognition-prediction-wearable/Classifiers/HMM/test/'
	
	sequence = 'Participant_909_Setup_A_Seq_3_Trial_4'

	list_participant = ['Participant_909']

	video_input = '/home/amalaise/Documents/These/experiments/AnDy-LoadHandling/annotation/Videos_Xsens/Participant_909/' + sequence + '.mp4'
	# video_input = 'video_reco3.mp4'


	print('Loading data...')

	path_wrapper = '/home/amalaise/Documents/These/experiments/ANDY_DATASET/AndyData-lab-onePerson/'


	# best_features = find_best_features('wrapper_dimensions_' + name_track + ".csv")
	file_name = path_wrapper + 'wrapper_'

	if(name_track == 'current_action'):
		best_features = tools.get_best_features(file_name + name_track + '.csv_11')[0]
	else:
		best_features = tools.get_best_features(file_name + name_track + '.csv_5')[0]

	dim_features = np.ones(len(best_features))

	print(best_features)

	timestamps = []
	data_win2 = []
	real_labels = []
	list_states = []

	info_participant = []
	info_sequences = []
	i=0



	for participant, nbr in zip(list_participant, range(len(list_participant))):
		path_data = '/home/amalaise/Documents/These/experiments/ANDY_DATASET/AndyData-lab-onePerson/xsens/allFeatures_csv/Participant_909/'
		print('Loading: ' + participant)
		
		list_files = ['Participant_909_Setup_A_Seq_3_Trial_4.csv']

		for file in list_files:
			name_seq = os.path.splitext(file)[0]

			info_participant.append(participant)
			info_sequences.append(name_seq)

			

			ref_data = DataBase(path + '/' + participant, name_seq)

			data, labels, time, list_s, list_features = tools.load_data(path, participant, name_seq, name_track, labels_folder)


			data_base = pd.DataFrame(data, columns = list_features)

			# time = data_base['timestamp']
		
			# labels, states = ref_data.load_labels_refGT(time, name_track, 'labels_3A')
			# ref_data.load_labels_ref(name_track, labels_folder)
			# labels = ref_data.get_real_labels(time)
			# states = ref_data.get_list_states()

			real_labels.append(labels)
			data_win2.append(data_base[best_features].as_matrix())
			timestamps.append(time)

			
			i += 1

	model = ModelHMM()
	model.load_model(path_model + 'test_' + name_track)
	list_states = model.get_list_states()

	print(list_states)


	predict_labels, proba = model.test_model(data_win2)

	real_labels = np.asarray(real_labels).T
	predict_labels = np.asarray(predict_labels).T

	

	# for i in range(len(predict_labels)):
	# 	print(predict_labels[i][0])
	# 	predict_labels[i][0] = predict_labels[i][0].replace("Id", "Idle")
	# 	predict_labels[i][0] = predict_labels[i][0].replace("Re", "Reach")
	# 	predict_labels[i][0] = predict_labels[i][0].replace("Rl", "Release")
	# 	predict_labels[i][0] = predict_labels[i][0].replace("Fm", "Fine Manipulation")
	# 	predict_labels[i][0] = predict_labels[i][0].replace("Sc", "Screw")
	# 	predict_labels[i][0] = predict_labels[i][0].replace("Ca", "Carry")
	# 	predict_labels[i][0] = predict_labels[i][0].replace("Pl", "Place")
	# 	predict_labels[i][0] = predict_labels[i][0].replace("Pi", "Pick")
	# 	predict_labels[i][0]

	
	# v_tools.video_sequence(real_labels, predict_labels, video_input, video_name)
	v_tools.video_distribution(proba[0], list_states, real_labels, 8, '', 'test_distribution_' + name_track + '2')
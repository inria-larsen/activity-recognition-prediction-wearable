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

list_joints = ['jL5S1',
			'jL4L3',
			'jL1T12',
			'jT9T8',
			'jT1C7',
			'jC1Head',
			'jRightT4Shoulder',
			'jRightShoulder',
			'jRightElbow',
			'jRightWrist',
			'jLeftT4Shoulder',
			'jLeftShoulder',
			'jLeftElbow',
			'jLeftWrist',
			'jRightHip',
			'jRightKnee',
			'jRightAnkle'
			]

list_segments = ['Pelvis',
			'L5',
			'L3',
			'T12',
			'T8',
			'Neck',
			'Head',
			'RightShoulder',
			'RightUpperArm',
			'RightForeArm',
			'RightHand',
			'LeftShoulder',
			'LeftUpperArm',
			'LeftForeArm',
			'LeftHand',
			'RightUpperLeg',
			'RightLowerLeg',
			'RightFoot',
			'RightToe',
			'LeftUpperLeg',
			'LeftLowerLeg',
			'LeftFoot',
			'LeftToe']


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

	nbr_features = 50

	name_feature = 'jointAngle_'
	name_feature = 'position_'

	list_features_final = []

	# for joint in list_joints:
	# 	list_features_final.append(name_feature + joint + '_' + 'x')
	# 	list_features_final.append(name_feature + joint + '_' + 'y')
	# 	list_features_final.append(name_feature + joint + '_' + 'z')

	for segment in list_segments:
		list_features_final.append(name_feature + segment + '_' + 'x')
		list_features_final.append(name_feature + segment + '_' + 'y')
		list_features_final.append(name_feature + segment + '_' + 'z')

	# list_features_final = ['comVel_centerOfMass_x', 'comVel_centerOfMass_y', 'comVel_centerOfMass_z', 'velocityNorm']
	
	# list_participant = ['541', '909', '3327', '5124', '5521', '5535', '8410', '9266', '9875']
	# list_participant = ['Participant_541']

	# # path_video = '/home/amalaise/Documents/These/experiments/AnDy-LoadHandling/annotation/Videos_Xsens/Participant_541/Participant_541_Setup_A_Seq_3_Trial_1.mp4'


	print('Loading data...')

	timestamps = []
	data_win2 = []
	real_labels = [[],[],[],[]]
	list_states = [[], [], [], []]

	info_participant = []
	info_sequences = []

	tracks = ['general_posture', 'detailed_posture', 'details', 'current_action']

	path_annotation = '/home/amalaise/Documents/These/experiments/ANDY_DATASET/AndyData-lab-onePerson/annotations/labels_csv2/'

	for participant, nbr in zip(list_participant, range(len(list_participant))):
		path_data = path_data_root  + participant
		print('Loading: ' + participant)
		
		list_files = os.listdir(path_data)
		list_files.sort()


		for file in list_files:
			name_seq = os.path.splitext(file)[0]
			
			data, labels, time, list_s, list_features = tools.load_data(path, participant, name_seq, 'general_posture', labels_folder)
			data_win2.append(data)

			# data_process = np.zeros([len(data), 8])
			# data_process[:,0:4] = data[:,0:4]
			# k = 4
			# for j in range(4, 14, 3):
			# 	data_process[:, k] =  np.linalg.norm(data[:, j:j+3], axis = 1)
			# 	k += 1

			# list_features_final = ['comVel_centerOfMass_x', 'comVel_centerOfMass_y', 'comVel_centerOfMass_z',
			# 	'velocityNorm', 'velocity_RightHandNorm',
			# 	'velocity_LeftHandNorm', 'velocity_LeftFootNorm', 'velocity_RightFootNorm']

			for name_track, num_track in zip(tracks, range(len(tracks))):
				labels, states = tools.load_labels_ref(time, path_annotation + participant + '/' + name_seq + '.labels.csv',
					name_track, participant, 1)
				real_labels[num_track].append(labels)

				for state in states:
					if(state not in list_states[num_track]):
						list_states[num_track].append(state)
						list_states[num_track] = sorted(list_states[num_track])

		
			# df_all_data = pd.DataFrame(data, columns = list_features)

			

			# timestamps.append(time)

			

	print(list_states)

	for name_track, num_track in zip(tracks, range(len(tracks))):

		F1_score = []
		dim_score = []
		feaures_save = []
		dim_features = np.ones(len(list_features_final))

		for n_iter in range(nbr_cross_val):
			print(n_iter)
			data_ref1, labels_ref, data_test, labels_test, id_train, id_test = tools.split_data_base2(data_win2, real_labels[num_track], ratio)

			data_ref = []
			for data in data_ref1:
				df = pd.DataFrame(data)
				df.columns = list_features
				data_ref.append(df[list_features_final].values)

			print('aaa', np.shape(data_ref))
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
			dim_score.append(8)
			feaures_save.append(str(list_features_final))

		print(np.shape(dim_score), np.shape(F1_score), np.shape(feaures_save))
		score_totaux = pd.DataFrame(
		{'nbr_components': dim_score,
		 'score': F1_score,
		 'features': feaures_save
		})

		score_totaux.to_csv('score_position' + '_' + name_track + ".csv", index=False)




	# plt.show()
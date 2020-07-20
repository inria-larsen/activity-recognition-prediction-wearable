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
	local_features = 1

	# list_participant = ['Participant_541']


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

		for n_components in range(1, 16):
			for n_iter in range(nbr_cross_val):
				data_ref, labels_ref, data_test, labels_test, id_train, id_test = tools.split_data_base2(data_win2, real_labels[num_track], ratio)

				obs = data_ref[0]
				labels = real_labels[0]

				for i in range(1, len(data_ref)):
					obs = np.concatenate([obs, data_ref[i]])

				x = StandardScaler().fit_transform(obs)
				pca = PCA(n_components=n_components)
				principalComponents = pca.fit_transform(x)

				col = []
				dim_features = []
				for i in range(n_components):
					col.append('col' + str(i))
					dim_features.append(1)

				for id_train in range(len(data_ref)):
					data_ref[id_train] = pca.transform(data_ref[id_train])

				for id_test in range(len(data_test)):
					data_test[id_test] = pca.transform(data_test[id_test])

				model = ModelHMM()
				model.train(data_ref, labels_ref, col, dim_features)

				pred_labels, proba = model.test_model(data_test)

				F1_temp = []
				for i in range(len(labels_test)):
					F1_temp.append(tools.compute_F1_score(labels_test[i], pred_labels[i], list_states[num_track]))

				F1_score.append(np.mean(F1_temp))
				dim_score.append(n_components)

					
				score_totaux = pd.DataFrame(
				{'nbr_components': dim_score,
				 'score': F1_score,
				})

				score_totaux.to_csv('score/score_pca' + '_' + name_track + "local.csv", index=False)




	# plt.show()



# from hmm_model import ModelHMM
# from data_base import DataBase
# import data_processing as pr
# import numpy as np
# import matplotlib.pyplot as plt
# import sys
# import yarp
# import visualization_tools as v_tools
# import tools
# import pandas as pd 
# from copy import deepcopy
# from mem_top import mem_top
# from sys import getsizeof
# import os
# import re
# from scipy import stats
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA

# import warnings
# warnings.filterwarnings("ignore")


# if __name__ == '__main__':
# 	# Prepare the data base
# 	yarp.Network.init()
# 	rf = yarp.ResourceFinder()
# 	rf.setDefaultContext("online_recognition")
# 	rf.setDefaultConfigFile("default.ini")
# 	rf.configure(sys.argv)

# 	path = rf.find('path_data_base').toString()
# 	path_model = rf.find('path_model').toString()
# 	name_model = rf.find('name_model').toString()
# 	name_track = rf.find('level_taxonomy').toString()
# 	labels_folder = rf.find('labels_folder').toString()

# 	list_participant = os.listdir(path)
# 	list_participant.sort()
# 	print(list_participant)

# 	# list_participant = ['5124']
# 	participant_label = []
# 	num_sequence = []
# 	testing = 1
# 	save = 0
# 	ratio = [70, 30, 0]
# 	nbr_cross_val = 10
# 	test_generalisation = 0
# 	method = 'wrapper'
# 	test_iteration = 1

# 	nbr_features = 50

# 	nbr_component = 50
	
# 	# # path_video = '/home/amalaise/Documents/These/experiments/AnDy-LoadHandling/annotation/Videos_Xsens/Participant_541/Participant_541_Setup_A_Seq_3_Trial_1.mp4'


# 	# id_test = 0

# 	print('Loading data...')

# 	timestamps = []
# 	data_win2 = []
# 	real_labels = []
# 	list_states = []

# 	info_participant = []
# 	info_sequences = []
# 	i=0

# 	for participant, nbr in zip(list_participant, range(len(list_participant))):
# 		path_data = path + '/' + participant + '/data_csv/'
# 		print('Loading: ' + participant)
		
# 		list_files = os.listdir(path_data)
# 		list_files.sort()

# 		for file in list_files:
# 			name_seq = os.path.splitext(file)[0]

# 			info_participant.append(participant)
# 			info_sequences.append(name_seq)

# 			data_base = pd.read_csv(path_data + file)
# 			ref_data = DataBase(path + '/' + participant, name_seq)

# 			list_features = list(data_base.columns.values)
# 			del list_features[0:2]
# 			dim_features = np.ones(len(list_features))

# 			time = data_base['timestamps']

# 			labels, states = ref_data.load_labels_refGT(time, name_track, 'labels_3A')

# 			real_labels.append(labels)
# 			data_win2.append(data_base[list_features].as_matrix())
# 			timestamps.append(time)

			

# 			for state in states:
# 				if(state not in list_states):
# 					list_states.append(state)
# 					list_states = sorted(list_states)
# 			i += 1

# 	print(list_states)

# 	F1_score = []
# 	dim_score = []
# 	for n_components in range(1, nbr_component):
# 		print('#############################################')
# 		print('iter:', n_components)	
# 		print('#############################################')

# 		for n_iter in range(nbr_cross_val):
# 			print(n_iter)
# 			data_ref, labels_ref, data_test, labels_test, id_train, id_test = tools.split_data_base2(data_win2, real_labels, ratio)

# 			obs = data_ref[0]
# 			labels = real_labels[0]

# 			for i in range(1, len(data_ref)):
# 				obs = np.concatenate([obs, data_ref[i]])

# 			x = StandardScaler().fit_transform(obs)
# 			pca = PCA(n_components=n_components)
# 			principalComponents = pca.fit_transform(x)

# 			col = []
# 			dim_features = []
# 			for i in range(n_components):
# 				col.append('col' + str(i))
# 				dim_features.append(1)

# 			for id_train in range(len(data_ref)):
# 				data_ref[id_train] = pca.transform(data_ref[id_train])

# 			for id_test in range(len(data_test)):
# 				data_test[id_test] = pca.transform(data_test[id_test])

# 			model = ModelHMM()
# 			model.train(data_ref, labels_ref, col, dim_features)

# 			pred_labels, proba = model.test_model(data_test)

# 			F1_temp = []
# 			for i in range(len(labels_test)):
# 				F1_temp.append(tools.compute_F1_score(labels_test[i], pred_labels[i], list_states))

# 			F1_score.append(np.mean(F1_temp))
# 			dim_score.append(n_components)



# 		score_totaux = pd.DataFrame(
# 		{'nbr_components': dim_score,
# 		 'score': F1_score,
# 		})

# 		score_totaux.to_csv('score_pca' + '_' + name_track + "2.csv", index=False)
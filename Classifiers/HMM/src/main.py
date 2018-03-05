from hmm_model import ModelHMM
from data_base import DataBase
import data_processing as pr
import numpy as np
import matplotlib.pyplot as plt
import sys
import yarp
import visualization_tools as v_tools
import tools

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
	# Prepare the data base
	name_track = 'detailed_posture'

	yarp.Network.init()
	rf = yarp.ResourceFinder()
	rf.setDefaultContext("online_recognition")
	rf.setDefaultConfigFile("default.ini")
	rf.configure(sys.argv)

	path = rf.find('path_data_base').toString()
	path_model = rf.find('path_model').toString()
	name_model = rf.find('name_model').toString()

	list_participant = ['909', '5521', '541', '3327']
	testing = 0
	# list_participant = ['5521', '541', '3327']


	# path_video = '/home/amalaise/Documents/These/experiments/AnDy-LoadHandling/annotation/Videos_Xsens/Participant_541/Participant_541_Setup_A_Seq_3_Trial_1.mp4'
	path_video = '/home/amalaise/Documents/These/experiments/AnDy-LoadHandling/annotation/Videos_Xsens/Participant_909/Participant_909_Setup_A_Seq_1_Trial_1.mp4'


	id_test = -1

	data_base = []

	for participant, nbr in zip(list_participant, range(len(list_participant))):

		data_base.append(DataBase(path + '/' + participant))
		data_base[nbr].load_mvnx_data()


		signals = rf.findGroup("Signals").tail().toString().replace(')', '').replace('(', '').split(' ')
		nb_port = int(len(signals))
		nb_active_port = 0

		list_features, dim_features = data_base[nbr].add_signals_to_dataBase(rf)

		# if('eglove' in signals):
		# 	info_signal = rf.findGroup('eglove')
		# 	list_items = info_signal.findGroup('list').tail().toString().split(' ')
		# 	data_glove, time_glove = data_base[nbr].add_data_glove(list_items)


	
	timestamps = []
	data_win = []
	data_glove2 = []
	timestamps_glove = []
	real_labels = []
	list_states = []

	window_size = float(rf.find('slidding_window_size').toString())
	print(list_features)
	for j in range(len(data_base)):
		if(j == id_test):
			continue

		sub_data = pr.concatenate_data(data_base[j], list_features)
		n_seq = data_base[j].get_nbr_sequence()

		t = []

		for i in range(n_seq):
			t.append(data_base[j].get_data_by_features('time', i))


			data_out, timestamps_out = pr.slidding_window(sub_data[i], t[i], window_size)
			t[i] = timestamps_out
			data_win.append(data_out)
			timestamps.append(t)

		data_base[j].load_labels_ref(name_track)
		labels = data_base[j].get_real_labels(t)
		for seq_labels in labels:
			real_labels.append(seq_labels)


		states = data_base[j].get_list_states()

		for state in states:
			if(state not in list_states):
				list_states.append(state)
				list_states = sorted(list_states)

		# data_out, timestamps_out = pr.slidding_window(data_glove[i], time_glove[i], window_size)
		# data_glove2.append(data_out)
		# time_glove[i] = timestamps_out
		
						
		# data_out, timestamps_out = pr.slidding_window(sub_data[i], timestamps[i], window_size)
		# data_win[i] = np.concatenate((data_win[i], data_out), axis = 1)

	# for i in range(n_seq):
	# 	t_mocap = timestamps[0]
	# 	t_glove = time_glove[0]

	# 	mocap_data = data_win[0]
	# 	glove_data = data_glove2[0]

	# 	t = 0
	# 	while(t_glove[t] < t_mocap[0]):
	# 		t += 1
	# 	t_start = t

	# 	while(t_mocap[t] < t_glove[-1]):
	# 		t += 1
	# 	t_end = t-1


	# 	d_mocap = mocap_data[0:t_end]
	# 	dg.append(glove_data[t_start:-1])

	# 	tg.append(t_glove[t_start:-1])
	# 	tm = t_mocap[0:t_end]


	# plt.figure()

	# plt.plot(tm, d_mocap)

	# plt.plot(tg, d_glove)

	
	# list_features = ['eglove']
	# dim_features = [4]
	# dim_features = data_base.get_dimension_features(list_features_mod)
	# n_states = len(list_states)
	

# 	#############
	


	index_sequences = range(n_seq)
	# index_sequences = np.arange(1, n_seq, 1)


# 	# for i in range(len(data_win)):
# 	# 	obs = data_win[i]
# 	# 	vel.append(np.diff(obs, axis=0))


# 	# for feature in range(nbr_features):
# 	# 	list_features_mod.insert(feature*2+1, list_features_mod[feature*2] + '-std')
# 	# 	list_features_mod[feature*2] = list_features_mod[feature*2] + '-mean'
# 	# 	dim_features.insert(feature*2+1, dim_features[feature*2])

# ##############
	model = ModelHMM()
	model.train(data_win, real_labels, list_features, dim_features, index_sequences)
	# model.train(data_win, real_labels, list_features, dim_features, index_sequences)

	model.save_model(path_model, name_model, "load_handling")


	#### Test

	if(testing):

		sub_data = pr.concatenate_data(data_base[id_test], list_features)
		n_seq = data_base[id_test].get_nbr_sequence()
		timestamps = []
		data_win = []

		for i in range(n_seq):
			timestamps.append(data_base[id_test].get_data_by_features('time', i))
			data_out, timestamps_out = pr.slidding_window(sub_data[i], timestamps[i], window_size)
			data_win.append(data_out)
			# data_out, timestamps_out = pr.slidding_window(sub_data[i], timestamps[i], window_size)
			# data_win[i] = np.concatenate((data_win[i], data_out), axis = 1)
			timestamps[i] = timestamps_out


		data_base[id_test].load_labels_ref(name_track)
		real_labels = data_base[id_test].get_real_labels(timestamps)

	# 	# # for i in range(len(data_win)):
	# 	# i = 0
		obs = data_win[0]
	# 	# print(np.shape(obs))
	# 	# # 	vel = np.diff(obs, axis=0)
		results = model.predict_states(obs)
		plt.figure()
		# plt.plot(obs2)
		plt.plot(results)
		plt.plot(obs)
		y_axis = np.arange(0, len(list_states), 1)
		plt.yticks(y_axis, list_states)


		real_labels = data_base[id_test].get_ref_labels()[0]
		# real_index = []
		# for j in range(len(real_labels)):
		# 	real_index.append(list_states.index(real_labels[j]))

		# plt.plot(real_index)


		pred_labels = []
		for j in range(len(results)):
			pred_labels.append(list_states[results[j]])

		# print((tg[0][-1] - tg[0][0])/len(tg[0]))


		v_tools.video_sequence(real_labels, pred_labels, path_video, 'test.mp4')

		plt.show()



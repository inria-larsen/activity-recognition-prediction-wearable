from hmm_model import ModelHMM
from data_base import DataBase
import data_processing as pr
import numpy as np
import matplotlib.pyplot as plt
import sys
import yarp
import visualization_tools as v_tools
import tools


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

	data_base = DataBase(path)
	data_base.load_mvnx_data()


	signals = rf.findGroup("Signals").tail().toString().replace(')', '').replace('(', '').split(' ')
	nb_port = int(len(signals))
	nb_active_port = 0

	list_features = []

	for signal in signals:
		info_signal = rf.findGroup(signal)
		is_enabled = int(info_signal.find('enable').toString())

		if(is_enabled):
			list_part = info_signal.findGroup('list').tail().toString().split(' ')
			list_features, dim_features = data_base.add_mvnx_data([signal], list_part)


	data_base.load_labels_ref()

	print(data_base.get_list_features())

	# Pre-processing
	# list_features = ['jointAngle']
	
	sub_data = pr.concatenate_data(data_base, list_features)

	# plt.plot(sub_data[0])

	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')
	
	# # ax.set_xlim(-3, -1)

	# v_tools.draw_pos(ax, position[0])
	# ax.set_aspect('equal')

	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')
	
	# # ax.set_xlim(-3, -1)

	# print(np.shape(pose))

	# v_tools.draw_pos(ax, pose[0])
	# ax.set_aspect('equal')


	# sub_data = pr.concatenate_data(data_base, ['centerOfMass', 'position'])

	size_window = 60
	n_seq = data_base.get_nbr_sequence()
	data_win = pr.slidding_window(sub_data, size_window)



	# list_states = data_base.get_list_states()
	# dim_features = data_base.get_dimension_features(list_features)
	# n_states = len(list_states)
	
	time = pr.concatenate_data(data_base, ['time'])
	timestamps = pr.set_timestamps(time, size_window)
	real_labels = data_base.get_real_labels(timestamps)


	index_sequences = range(n_seq)

	vel = []

	# for i in range(len(data_win)):
	# 	obs = data_win[i]
	# 	vel.append(np.diff(obs, axis=0))

	model = ModelHMM()
	model.train(data_win, real_labels, list_features, dim_features, index_sequences)

	# for i in range(len(data_win)):
	i = 0
	obs = data_win[0]
	# 	vel = np.diff(obs, axis=0)
	results = model.predict_states(obs)
	plt.figure()
	plt.plot(obs)
	plt.plot(results)

	# plt.plot(np.linalg.norm(vel, axis = 1))

	model.save_model(path_model, name_model, "pick_and_place")

	# model.load_model(path_model + '/' + name_model)

	# mu, sigma = model.get_emission_prob()
	# print(mu)
	# print(np.sum(A, axis=1))



	# 
	plt.show()



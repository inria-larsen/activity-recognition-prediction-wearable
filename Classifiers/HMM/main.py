from src.hmm_model import ModelHMM
from src.data_base import DataBase
import src.data_processing as pr
import numpy as np
import matplotlib.pyplot as plt


path = '/home/amalaise/Documents/Xsens/pick_and_place/'

if __name__ == '__main__':
	# Prepare the data base
	data_base = DataBase(path)
	data_base.load_mvnx_data()
	data = data_base.add_mvnx_data(['centerOfMass'])
	data = data_base.add_features_by_segments('velocity', [ 'Head'])
	# data = data_base.add_features_by_joints('jointAngle', ['jL5S1'])
	# data = data_base.add_features_by_sensors('sensorAcceleration', ['Pelvis'])
	data_base.load_labels_ref()

	print(data_base.get_list_features())

	# Pre-processing
	# list_features = ['centerOfMass']
	list_features = ['centerOfMass', 'velocity_Head']
	sub_data = pr.concatenate_data(data_base, list_features)
	# sub_data = pr.concatenate_data(data_base, ['centerOfMass', 'position'])

	size_window = 60
	n_seq = data_base.get_nbr_sequence()
	data_com = pr.slidding_window(sub_data, size_window)



	list_states = data_base.get_list_states()
	dim_features = data_base.get_dimension_features(list_features)
	n_states = len(list_states)
	
	time = pr.concatenate_data(data_base, ['time'])
	timestamps = pr.set_timestamps(time, size_window)
	real_labels = data_base.get_real_labels(timestamps)


	index_sequences = range(n_seq)

	model = ModelHMM()
	model.train(data_com, real_labels, list_features, dim_features, index_sequences)

	# model.save_model(path_save, 'model', 'pick_and_place')

	# A = model.get_trans_mat()
	# print(A)
	# print(np.sum(A, axis=1))

	obs = data_com[0]
	results = model.predict_states(obs)

	plt.plot(results)
	plt.show()



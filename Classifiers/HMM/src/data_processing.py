import numpy as np
from data_base import DataBase



def selection_feature(data_base):

	return reduce_data_base

def slidding_window2(data_set, size_window):
	data_window = []
	n_seq = len(data_set)
	for seq in range(n_seq):
		win = int(size_window/2)
		data_out = []
		data = data_set[seq]
		for i in np.arange(win, len(data)-win, win):
			average = np.mean(data[i-win : i+win+1], axis=0)
			data_out.append(average)

		data_window.append(np.zeros(np.shape(data_out)))

		for i in range(len(data_out)):
			data_window[seq][i,:] = data_out[i]


	return data_window

def slidding_window(data_input, timestamps, size_window):
	tic_time = timestamps[0]
	buffer_data = []
	data_out = []

	timestamps_out = []

	flag = 0


	for frame in range(len(timestamps)):

		if((timestamps[frame] - tic_time) >= size_window/2):
			data_out.append(np.mean(buffer_data, axis = 0))
			length = int(len(buffer_data)/2)
			timestamps_out.append(tic_time)
			tic_time = timestamps[frame]
			if(flag == 0):
				flag = 1
			else:
				del buffer_data[0:length]
		else:
			buffer_data.append(np.asarray(data_input[frame]))

	data_window = np.zeros(np.shape(data_out))

	for i in range(len(data_out)):
		data_window[i,:] = data_out[i]

	return data_window, timestamps_out


def concatenate_data(data_set, list_features):	
	data_out = []
	index = 0
	for features in list_features:
		data_in = data_set.get_data_by_features(features)

		if(len(np.shape(data_in)) == 1):
			data_in = np.expand_dims(data_in, axis=1)
		if(index == 0):	
			data_out = data_in
		else:
			data_out = np.concatenate((data_out, data_in), axis = 1)
		index += 1

	return data_out


def normalization(data_set):
	return data_set

def set_timestamps(timestamps, size_window):
	n_seq = len(timestamps)
	time = [[]]
	for seq in range(n_seq):
		win = int(size_window/2)
		for i in np.arange(win, len(timestamps[seq])-win, win):
			time[seq].append(timestamps[seq][i])
		if(seq < n_seq - 1):
			time.append([])
	return time


def diff(self, data, frequence):
	T, N = np.shape(data)
	data_diff = np.zeros((T, N))
	for t in range(1, T):
		data_diff[t, :] = (data[t][:] - data[t-1][:])*frequence
	return data_diff
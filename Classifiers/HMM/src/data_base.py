from xsens_parser import mvnx_tree
from anvil_parser import anvil_tree
from eglove_parser import eglove_tree
import numpy as np
import os
import scipy.signal
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
import csv
from collections import Counter
import random 

segment_features = [
			'orientation',
			'position',
			'velocity',
			'acceleration',
			'angularVelocity',
			'angularAcceleration'
					]

joint_features = [
			'jointAngle',
			'jointAngleXZY' #,
			#'angularVelocity',
			#'angularAcceleration'
					]

sensor_features = [
			'sensorOrientation',
			'sensorAngularVelocity',
			'sensorAcceleration'
					]

features_mvnx = [ 
				'orientation',
				'position',
				'velocity',
				'acceleration',
				'angularVelocity',
				'angularAcceleration',
				'jointAngle',
				'jointAngleXZY',
				'centerOfMass',
				'sensorOrientation',
				'sensorAngularVelocity',
				'sensorFreeAcceleration'
				]



class DataBase():
	""" A class of the data base containing the data from xsens sensors and labels

	The constructor of a DataBase instance needs a path that contains the data folders.
	The database must be in a folder wich contains subfolder for each category of data:
	Each folder must have the same number of files corresponding to each sequence
	mvnx: folder containing the mvnx data (xml) from xsens for each sequence
	labels: contains the xml files of manual annotation with Anvil software for each sequence
	video: contains the video of the xsens model
	"""

	mvnx_folder = "/mvnx/"
	eglove_folder = "/glove/"
	videos_folder = "/video/"

	def __init__(self, path, file):
		"""
		Constructor of the dataset.
		Parameters;
		path: the path where is located the different folders
		n_seq (optional): the number of sequence to add in the dataset
		"""
		self.path_data = path
		self.name_seq = ''
		self.mvnx_tree = []
		self.mocap_data = []
		self.ref_data = []
		self.list_features = [[], []]
		self.list_states = []
		self.name_seq = file


	def load_mvnx_data(self, path):
		""" 
		Load the mocap mvnx files corresponding to xsens MoCap data
		"""
		self.mvnx_tree = mvnx_tree(path + self.name_seq + '.mvnx')
		self.mocap_data.append(self.mvnx_tree.get_timestamp_ms())

		# set the timestamp as the first data on the list of features
		self.list_features[0].append('time')
		self.list_features[1].append(1)
		return


	def load_labels_ref(self, name_track = '0', labels_folder = 'labels'):
		""" Load the reference labels from Anvil file

		This function updates the reference data and the list of states.
		The reference data is a list of dimension n_sequence containing for each three lists:
		[0]: label of the activities
		[1]: relative timestamps in ms of the start of activities
		[2]: relative timestamps in ms of the end of activities

		list_states: sorted list of string containing the list of states
		"""
		path = self.path_data + '/' + labels_folder + '/'
		time_start = [[]]

		self.ref_data = []
		self.real_labels = []
		self.list_states = []

		ref = anvil_tree(path + self.name_seq + '.anvil')
		labels, start, end = ref.get_data(name_track)

		for i in range(len(labels)):
			if labels[i] == 'NONE':
				labels[i] = labels[i+1]
			# labels[i] = labels[i].replace("Kn","Cr")

		self.ref_data.append([labels, start, end])
			
		for state in sorted(ref.get_list_states()):
			if(state not in self.list_states):
				self.list_states.append(state)
				self.list_states = sorted(self.list_states)

		return

	def load_labels_refGT(self, timestamps, name_track, labels_folder = 'labels'):
		""" Load the reference labels from csv file from ground truth

		This function updates the reference data and the list of states.
		Take in input the timestamps corresponding to the motion capture data
		Take in input the name of the track used for the labels and the label folder

		list_states: sorted list of string containing the list of states
		"""
		path = self.path_data + '/' + labels_folder + '/'
		time_start = [[]]

		self.ref_data = []
		self.real_labels = []
		self.list_states = []

		self.real_labels = pd.read_csv(path + self.name_seq + '_GT.csv')[name_track].values
		self.list_states, l = np.unique(self.real_labels, return_inverse=True)

		return self.real_labels, self.list_states

		

	def load_labels_ref3A(self, timestamps, name_track, participant, GT = 0):
		""" Load the reference labels from csv file from 3 annotators

		This function updates the reference data and the list of states.
		Take in input the timestamps corresponding to the motion capture data
		Take in input the name of the track used for the labels and the label folder
		Possibility to extract the Ground Truth if GT = 1 (default = 0)

		list_states: sorted list of string containing the list of states
		"""

		timestamps = np.asarray(timestamps)
		path = self.path_data + '/annotations/labels_csv2/' + participant + '/'
		time_start = [[]]

		self.ref_data = []
		self.real_labels = []
		self.list_states = []

		columns_labels = ['t_video']
		for i in range(3):
			columns_labels.append(name_track + '_Annotator' + str(i+1))


		df_labels = pd.read_csv(path + self.name_seq +  '.labels.csv')
		df_labels = df_labels[columns_labels]

			# t_sample = (timestamps - timestamps[0]).tolist()

		t_sample = df_labels['t_video']
		# t_sample.append(timestamps[-1])

		count = 0
		labels = []
		for t in t_sample:
			labels.append(df_labels.iloc[count, 1:4].values)
			time = df_labels.iloc[count, 0]
			if(t > time):
				count += 1
				if(count == len(df_labels)):
					break

		labels = np.array(labels)

		timestamps = timestamps - timestamps[0]
		# timestamps = np.array(df_labels['t_video'])

		# sliding window 
		T = 0.25

		# t_sample = pd.DataFrame({'t': t_sample})
		df_labels = pd.DataFrame(labels)

		if(GT):
			for i in range(0, len(timestamps)):
				# if(timestamps[i]+T/2 >= timestamps[-1]):
				# 	break
				df_labels_win = df_labels[t_sample.between(timestamps[i], timestamps[i]+T, inclusive=True)]
				if(df_labels_win.empty):
					break
				list_label_win = np.concatenate(df_labels_win.values)
				cnt = Counter(list_label_win)
				if(len(cnt) >= 2):
					if(cnt.most_common(2)[0][1] == cnt.most_common(2)[1][1]):
						list_label_win = []
						for j in range(len(df_labels_win)):
							cnt = Counter(df_labels_win.iloc[j].values.tolist())
							if(len(cnt) >= 2):
								if(cnt.most_common(2)[0][1] == cnt.most_common(2)[1][1]):
									list_label_win.append('NONE')
								else:
									list_label_win.append(cnt.most_common(1)[0][0])
							else:
								list_label_win.append(cnt.most_common(1)[0][0])

						cnt = Counter(list_label_win)

				self.real_labels.append((cnt.most_common(1)[0][0]))

		else:
			self.real_labels = df_labels

		seq_none = [[]]
		count_none = 0
		for i in range(0, len(self.real_labels)):   

			if(GT):
				if(self.real_labels[i]=='NONE'):
					seq_none[count_none].append(i)
					if(self.real_labels[i+1] != 'NONE'):
						count_none += 1
						seq_none.append([])

		del seq_none[-1]

		if(GT):
			for i in range(len(seq_none)):
				A = deepcopy(seq_none[i])
				while(len(seq_none[i]) >= 1):
					if(len(seq_none[i]) == 1):
						if(random.randint(0,1)):
							self.real_labels[seq_none[i][0]] = self.real_labels[seq_none[i][0]+1]
						else:
							self.real_labels[seq_none[i][0]] = self.real_labels[seq_none[i][0]-1]
						del seq_none[i][0]

					else:
						self.real_labels[seq_none[i][0]] = self.real_labels[seq_none[i][0]-1]
						self.real_labels[seq_none[i][-1]] = self.real_labels[seq_none[i][-1]+1]
						del seq_none[i][0]
						del seq_none[i][-1]

		self.list_states, l = np.unique(self.real_labels, return_inverse=True)

		return self.real_labels, self.list_states

	def add_signals_to_dataBase(self, config, processing=False):
		"""
		 This funciton aims to load the data from the mvnx file based on the .ini (rf) file in input
		"""
		self.config_file = config

		signals = config["DEFAULT"]["list_signals"].split(',')
		nb_port = int(len(signals))
		nb_active_port = 0

		for signal in signals:
			info_signal = config[signal]
			is_enabled = int(info_signal["enable"])


			related_data = info_signal['related_data']
			if(related_data == 'NONE'):
				related_data = signal

			# Retrieve the data from .ini file
			info_related_data = config[related_data]

			
			# Check if the signal is enabled
			if(is_enabled):
				list_items = info_signal['list'].split(',')
				# order_diff = info_signal.find('diff_order').toString()
				# is_norm = info_signal.find('norm').toString()
				# normalize = info_signal.find('normalize').toString()
				# dist_com = info_signal.find('dist_com').toString()

				# if(order_diff == 'NONE'):
				# 	order_diff = 0
				# else:
				# 	order_diff = int(order_diff)

				# if(is_norm == 'NONE'):
				# 	is_norm = 0
				# else:
				# 	is_norm = int(is_norm)

				#used to normalize the COM
				if(normalize == 'NONE'):
					normalize = 0
				else:
					normalize = int(normalize)

				# if(dist_com == 'NONE'):
				# 	dist_com = 0
				# else:
				# 	dist_com = int(dist_com)

				if(related_data in features_mvnx):
					data = self.mvnx_tree.get_data(related_data)

				# If processing = True, the position data are processed with transformation to 
				# be in relative position regarding pelvis
				if(related_data == 'position' and processing == True):

					# IMPORTANT
					# all this part is to transform the reference of the cartesian positions
					# wrt to the pelvis and not to the absolute world frame of the xsens
					# e.g. we want to have the hand position wrt to the pelvis
					# and for tall the segments as well
					# this transformation must be done also in sensor_processing to have the same
					# online values
					orientation = self.mvnx_tree.get_data('orientation')
					for t in range(len(data)):
						abs_data = deepcopy(data[t])
						o_data = orientation[t]

						# get pevis orientation
						i = 0
						q0 = o_data[i*4]
						q1 = o_data[i*4 + 1]
						q2 = o_data[i*4 + 2]
						q3 = o_data[i*4 + 3]

						# compute pelvis orientation matrix using quaternion orinetation matrix 
						# cf. xsens manual page 123, section 23.2.2
						R = np.array([[q0*q0 + q1*q1 - q2*q2 - q3*q3, 2*q1*q2 - 2*q0*q3, 2*q1*q3 + 2*q0*q2],
							[2*q1*q2 + 2*q0*q3, q0*q0 - q1*q1 + q2*q2 - q3*q3, 2*q2*q3 - 2*q0*q1],
							[2*q1*q3 - 2*q0*q2, 2*q2*q3 + 2*q0*q1, q0*q0 - q1*q1 - q2*q2 + q3*q3]])

						# TO FIX LATER: this should be from 1 to 23 -> range(1,23)
						for i in range(0, 23):
							# compute distance position wrt the pelvis position (pelvis is zero)
							data[t, i*3:i*3+3] = abs_data[i*3:i*3+3] - abs_data[0:3]
							data[t, i*3:i*3+3] = R@data[t,i*3:i*3+3]


				# Normalize the data regarding the initial position
				order_diff = 0
				if(normalize == 1):				
					data_normalize = deepcopy(data)
					data_init = deepcopy(data_normalize[0])
					for t in range(len(data_normalize)):
						data_normalize[t] = (data_normalize[t] - data_init)/data_init

					data = data_normalize


				# Derive the data depending on the order
				if(order_diff > 0):
					data_diff = np.diff(np.asarray(data), order_diff, axis=0)
					for j in range(order_diff):
						data_diff = np.insert(data_diff, 0, 0, axis=0)

					order = 6
					fs = 240
					nyq = 0.5 * fs
					cutoff = 30
					normal_cutoff = cutoff / nyq

					b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
					data = scipy.signal.lfilter(b, a, data_diff, axis = 0)*fs

				dimension = np.shape(data)[1]

				if(list_items[0] == ''):
					if(signal == 'CoMVelocityNorm'):
						data = data[:,0:2]	
					if(is_norm):
						data = np.linalg.norm(data, axis = 1)
						dimension = 1

					self.mocap_data.append(data)
					
					if(not(signal in self.list_features[0])):
						# Update the list of features with name and dimension
						self.list_features[0].append(signal)
						self.list_features[1].append(dimension)

				else:
					related_items = info_related_data["related_items"]

					if(list_items[0] == 'all'):
						items = []
						for key in config[related_items]: items.append(key)
						del items[0:2]
						del items[-7::]

					else: 
						items = deepcopy(list_items)

					for item in items:
						nb_items = int(config[related_items]["Total"])
						dimension_reduce = int(dimension/nb_items)

						id_item = config[related_items][item]
						id_item = int(id_item)*dimension_reduce

						data_reduce = data[:,id_item:id_item + dimension_reduce]

						# Vector data in one dimension
						# if(is_norm):
						# 	data_reduce = np.linalg.norm(data_reduce, axis = 1)
						# 	dimension_reduce = 1

						if(dimension_reduce > 1):
							list_quaternion = ['q0', 'q1', 'q2', 'q3']
							list_dimension = ['x', 'y', 'z']

							for d in range(dimension_reduce):
								if(signal == 'orientation' or signal == 'sensorOrientation'):
									name_dimension = list_quaternion[d]
								elif(signal == 'centerOfMass'):
									name_signal = signal + '_' + list_dimension[d]
									self.list_features[0].append(name_signal)
									self.list_features[1].append(1)
									self.mocap_data.append(data_reduce[:,d])
									continue
								else:
									name_dimension = list_dimension[d]

								self.mocap_data.append(data_reduce[:,d])
								name_signal = signal + '_' + item + '_' + name_dimension

								if(not(name_signal in self.list_features[0])):
									# Update the list of features with name and dimension
									self.list_features[0].append(name_signal)
									self.list_features[1].append(1)
									# self.list_features[1].append(dimension_reduce)

						else:
							self.mocap_data.append(data_reduce)


		list_features = self.list_features[0][1:]
		dim_features = self.list_features[1][1:]

		return list_features, dim_features


	def get_data_by_features(self, name_feature):
		"""
		Input: a string for the name of the feature
		Return the data corresponding to the feature in input
		"""
		index_feature =  self.list_features[0].index(name_feature)
		return self.mocap_data[index_feature]

	def get_dimension_features(self, list_features):
		dimension_list = [] 
		for feature in list_features:
			index_feature =  self.list_features[0].index(feature)
			dimension_list.append(self.list_features[1][index_feature])
		return dimension_list
		

	def get_real_labels(self, timestamps):
		""" Return the real labels related to the timestamps in input
		
		"""
		self.real_labels = []

		flag = 0
		time = (np.asarray(timestamps) - timestamps[0])

		end = self.ref_data[0][2]
		labels = self.ref_data[0][0]

		for t in time:
			if(t >= end[flag]):
				flag += 1
				if(flag >= len(end)):
					break
			self.real_labels.append(labels[flag])

		return self.real_labels



	def get_list_features(self):
		return self.list_features

	def get_mocap_data(self):
		return self.mocap_data

	def get_ref_data(self):
		return self.ref_data

	def get_ref_labels(self):
		return self.real_labels

	def get_nbr_sequence(self):
		return self.n_seq

	def get_list_states(self):
		return self.list_states

	def get_data_base(self):
		return self


	def add_data_glove(self, info_signal):
		"""
		Add data of eglove and synchronised them with motion capture timestamps
		"""
		path = self.path_data + self.eglove_folder
		list_files = os.listdir(path)
		list_files.sort()

		list_items = info_signal.findGroup('list').tail().toString().split(' ')

		self.glove_timestamps = []
		self.data_glove = []

		is_norm = info_signal.find('norm').toString()
		if(is_norm == ''):
			is_norm = 0
		else:
			is_norm = int(is_norm)

		with open(path + '/' + self.name_seq + '_glove.txt', 'rt') as f:
			reader = csv.reader(f, delimiter=' ')

			data_forces = []
			data_angles = []

			for row in reader:
				data_forces.append(list(map(float, row[0:7])))
				self.glove_timestamps.append(float(row[7]))


			data_reduce = np.asarray(data_forces)

			if(is_norm):				
				data_reduce = np.linalg.norm(data_forces, axis = 1)
				data_reduce = np.expand_dims(data_reduce, axis=1)


			self.data_glove = data_reduce
				
		return self.data_glove, self.glove_timestamps

	def get_timestamps_glove(self):
		return self.glove_timestamps

	def get_data_glove(self):
		return self.data_glove

	def synchronize(self):
		for seq in range(self.n_seq):
			t_mocap = self.get_data_by_features('time', seq).tolist()
			t_glove = self.get_timestamps_glove[seq]

			mocap_data = data_win[0].tolist()
			glove_data = data_glove[0].tolist()


			t = 0
			while(t_glove[t] < t_mocap[0]):
				t += 1
			t_start = t


			while(t_mocap[t] < t_glove[-1]):
				t += 1
			t_end = t+1

			del glove_data[0:t_start]
			del t_glove[0:t_start]

			del mocap_data[t_end:]
			del t_mocap[t_end:]

			print(t_mocap[0], t_glove[0])
			print(t_mocap[-1], t_glove[-1])

			data_force = np.zeros((len(t_mocap), np.shape(glove_data)[1]))

			print(np.shape(data_force), np.shape(glove_data))


			count = 0

			for i in range(len(t_mocap)):
				data_force[i] = glove_data[count]
				if(t_glove[count] < t_mocap[i]):
					count += 1
					if(count == len(glove_data)):
						break
		return


		# start = 0

		# for i in range(len(glove_timestamp)):
		# 	glove_timestamp[i] = float(glove_timestamp[i]) + 1
		# 	# t_glove[i] = float(t_glove[i])

		# for i in range(len(mocap_timestamp)):
		# 	mocap_timestamp[i] = float(mocap_timestamp[i])

		# start = 0
		# while(float(glove_timestamp[start]) < float(mocap_timestamp[0])):
		# 	start += 1

		# end = start

		# while(float(glove_timestamp[end]) <= float(mocap_timestamp[-1])):
		# 	end += 1
		# 	if(end == len(glove_timestamp)):
		# 		end -= 1
		# 		break

		# data_forces = data[start:end]
		# data_force = np.zeros((len(self.time), 1))

		# count = 0
		# flag = 0
		# # print(len(self.time), len(data_forces))
		# for i in range(len(self.time)-1):
		# 	data_force[i] = data_forces[count]
		# 	flag += 1
		# 	# if(flag == np.ceil(len(self.time) / len(data_forces))):
		# 	if(flag == 24):
		# 		count += 1
		# 		flag = 0
		# 		if(count == len(data_forces)):
		# 			break
		# 		# print(count, i)

		# self.all_data.append(data_force)
		# self.list_features[0].append('contactForces')
		# self.list_features[1].append(1)






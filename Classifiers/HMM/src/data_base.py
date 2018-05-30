from xsens_parser import mvnx_tree
from anvil_parser import anvil_tree
from eglove_parser import eglove_tree
import numpy as np
import os
import scipy.signal
import matplotlib.pyplot as plt
from copy import deepcopy
import csv

segment_features = [
			'orientation',
			'position',
			'velocity',
			'acceleration'
					]

joint_features = [
			'jointAngle',
			'jointAngleXZY',
			'angularVelocity',
			'angularAcceleration'
					]

sensor_features = [
			'sensorOrientation',
			'sensorAngularVelocity',
			'sensorAcceleration'
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

	labels_folder = "/labels/"
	mvnx_folder = "/mvnx/"
	eglove_folder = "/glove/"
	videos_folder = "/video/"

	def __init__(self, path, n_seq = -1):
		"""
		Constructor of the dataset.
		Parameters;
		path: the path where is located the different folders
		n_seq (optional): the number of sequence to add in the dataset
		"""
		self.path_data = path
		self.mvnx_tree = []
		self.mocap_data = [[]]
		self.ref_data = [[]]
		self.list_features = [[], []]
		self.list_states = []
		self.n_seq = n_seq


	def load_mvnx_data(self):
		""" 
		Load the mocap mvnx files corresponding to xsens MoCap data
		"""
		path = self.path_data + self.mvnx_folder
		list_files = os.listdir(path)
		list_files.sort()

		if(self.n_seq == -1):
			self.n_seq = len(list_files)

		# check all the files in the mvnx folder
		for file, i in zip(list_files, range(self.n_seq)):
			self.mvnx_tree.append(mvnx_tree(path + file))
			self.mocap_data[i].append(self.mvnx_tree[i].get_timestamp_ms())

			if(i < self.n_seq - 1):
				self.mocap_data.append([])

		# set the timestamp as the first data on the list of features
		self.list_features[0].append('time')
		self.list_features[1].append(1)
		return path


	def load_labels_ref(self, name_track = '0'):
		""" Load the reference labels from Anvil file

		This function updates the reference data and the list of states.
		The reference data is a list of dimension n_sequence containing for each three lists:
		[0]: label of the activities
		[1]: relative timestamps in ms of the start of activities
		[2]: relative timestamps in ms of the end of activities

		list_states: sorted list of string containing the list of states
		"""


		path = self.path_data + self.labels_folder
		list_files = os.listdir(path)
		list_files.sort()

		time_start = [[]]


		self.ref_data = [[]]
		self.real_labels = [[]]
		self.list_states = []
		
		for file, i in zip(list_files, range(self.n_seq)):
			ref = anvil_tree(path + file)

			labels, start, end = ref.get_data(name_track)
			# for in ref.get_data():
			# 	self.ref_data[i]
			self.ref_data[i].append(ref.get_data(name_track))
			
			for state in sorted(ref.get_list_states()):
				if(state not in self.list_states):
					self.list_states.append(state)
					self.list_states = sorted(self.list_states)


			flag = 0
			time = self.get_data_by_features('time', i)

			for t in time:
				if(t >= end[flag]):
					flag += 1
					if(flag >= len(end)):
						break
				self.real_labels[i].append(labels[flag])

			if(i < self.n_seq -1):
				self.real_labels.append([])
				self.ref_data.append([])

		return

	def add_signals_to_dataBase(self, rf):
		self.config_file = rf

		signals = self.config_file.findGroup("Signals").tail().toString().replace(')', '').replace('(', '').split(' ')
		nb_port = int(len(signals))
		nb_active_port = 0

		for signal in signals:
			info_signal = self.config_file.findGroup(signal)
			is_enabled = int(info_signal.find('enable').toString())

			if(signal == 'eglove'):
			 	# list_items = info_signal.findGroup('list').tail().toString().split(' ')
			 	# self.add_data_glove(list_items)
			 	continue

			related_data = info_signal.find('related_data').toString()
			if(related_data == ''):
				related_data = signal


			info_related_data = self.config_file.findGroup(related_data)
			
			if(is_enabled):
				list_items = info_signal.findGroup('list').tail().toString().split(' ')
				order_diff = info_signal.find('diff_order').toString()
				is_norm = info_signal.find('norm').toString()
				normalize = info_signal.find('normalize').toString()
				dist_com = info_signal.find('dist_com').toString()

				if(order_diff == ''):
					order_diff = 0
				else:
					order_diff = int(order_diff)

				if(is_norm == ''):
					is_norm = 0
				else:
					is_norm = int(is_norm)

				if(normalize == ''):
					normalize = 0
				else:
					normalize = int(normalize)

				if(dist_com == ''):
					dist_com = 0
				else:
					dist_com = int(dist_com)

				
				features_mvnx = self.mvnx_tree[0].get_list_features()

				for i in range(self.n_seq):
					if(related_data in features_mvnx):
						data = self.mvnx_tree[i].get_data(related_data)

					if(normalize == 1):				
						data_normalize = deepcopy(data)
						data_init = deepcopy(data_normalize[0])
						for t in range(len(data_normalize)):
							# data_normalize[t] = data_normalize[t] - data_init
							data_normalize[t] = (data_normalize[t] - data_init)/data_init
							# data_normalize[t] = data_normalize[t]/data_init

						data = data_normalize


							
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

					if((list_items[0] == 'all') or (list_items[0] == '')):
						if(signal == 'CoMVelocityNorm'):
							data = data[:,0:2]	
						if(is_norm):
							data = np.linalg.norm(data, axis = 1)
							dimension = 1

						self.mocap_data[i].append(data)
						
						if(not(signal in self.list_features[0])):
							# Update the list of features with name and dimension
							self.list_features[0].append(signal)
							self.list_features[1].append(dimension)

					else:
						related_items = info_related_data.find("related_items").toString()

						for item in list_items:
							nb_items = int(rf.findGroup(related_items).find('Total').toString())
							dimension_reduce = int(dimension/nb_items)

							id_item = rf.findGroup(related_items).find(item).toString()
							id_item = int(id_item)*dimension_reduce

							if(item == 'jL5S1'):
								data_reduce = data[:,id_item + 2]
								data_reduce = np.expand_dims(data_reduce, axis=1)
								dimension_reduce = 1
							else:
								data_reduce = data[:,id_item:id_item + dimension_reduce]

							if(is_norm):
								data_reduce = np.linalg.norm(data_reduce, axis = 1)
								dimension_reduce = 1

							if(dist_com):
								data_com = self.mvnx_tree[i].get_data('centerOfMass')

								data_dist = np.zeros((len(data_com)))

								for t in range(len(data_com)):

									data_dist[t] = np.sqrt(
										# (data_reduce[t, 0] - data_com[t, 0]) * (data_reduce[t, 0] - data_com[t, 0]) +
										# (data_reduce[t, 1] - data_com[t, 1]) * (data_reduce[t, 1] - data_com[t, 1]) +
										(data_reduce[t, 2] - data_com[t, 2]) * (data_reduce[t, 2] - data_com[t, 2]))
								
								data_reduce = np.expand_dims(data_dist, axis=1)
								dimension_reduce = 1


							self.mocap_data[i].append(data_reduce)

							name_signal = signal + '_' + item

							if(not(name_signal in self.list_features[0])):
								# Update the list of features with name and dimension
								self.list_features[0].append(name_signal)
								self.list_features[1].append(dimension_reduce)


		list_features = self.list_features[0][1:]
		dim_features = self.list_features[1][1:]

		return list_features, dim_features


	def add_mvnx_data(self, list_features, sub_list):
		""" Add to the dataset the data corresponding to the list of features in input

		input:
		list_features: list of string corresponding to the features in mvnx file
		ouput: returns all the data per sequence in a list of array
		"""

		features_mvnx = self.mvnx_tree[0].get_list_features()
				
		for feature in list_features:
			if(feature in self.list_features[0]):  # Check if the feature is not already in the database
				continue

			if(feature in features_mvnx): # Check if the feature exists in mvnx file
				if(feature == 'centerOfMass'):
					for i in range(self.n_seq):
						com = self.mvnx_tree[i].get_data(feature)
						if(((sub_list[0] == 'all') or (sub_list[0] == ''))):
							self.mocap_data[i].append(com)
							continue
						if('z' in sub_list[0]):
							self.mocap_data[i].append(com[:,0])
						if('y' in sub_list[0]):
							self.mocap_data[i].append(com[:,1])
						if('x' in sub_list[0]):
							self.mocap_data[i].append(com[:,2])
						
					# if(((sub_list[0] == 'all') or (sub_list[0] == ''))):
					# 	self.list_features[0].append(feature)
					# 	self.list_features[1].append(3)
		
					# if('z' in sub_list[0]):		
					# 	self.list_features[0].append('COM_Z')
					# 	self.list_features[1].append(1)

					# if('y' in sub_list[0]):		
					# 	self.list_features[0].append('COM_Y')
					# 	self.list_features[1].append(1)

					# if('x' in sub_list[0]):		
					# 	self.list_features[0].append('COM_X')
					# 	self.list_features[1].append(1)

					# if('z' in sub_list):

				elif((sub_list[0] == 'all') or (sub_list[0] == '')):
					for i in range(self.n_seq):
						self.mocap_data[i].append(self.mvnx_tree[i].get_data(feature))

					# Update the list of features with name and dimension
					self.list_features[0].append(feature)
					self.list_features[1].append(len(self.mvnx_tree[i].get_data(feature).T))

				elif(feature in segment_features):
					self.add_features_by_segments(feature, sub_list)

				elif(feature in joint_features):
					self.add_features_by_joints(feature, sub_list)

				elif(feature in sensor_features):
					self.add_features_by_sensors(feature, sub_list)

			# elif(feature == 'CoMVelocity'):
			# 	for i in range(self.n_seq):
			# 		com = self.mvnx_tree[i].get_data('centerOfMass')
			# 		com_vel = np.diff(com, axis=0)
			# 		com_vel = np.insert(com_vel, 0, 0, axis=0)

			# 		self.mocap_data[i].append(com_vel)

			# 	self.list_features[0].append(feature)
			# 	self.list_features[1].append(3)








		list_features = self.list_features[0][1:]
		dim_features = self.list_features[1][1:]

		return list_features, dim_features


	def add_features_by_segments(self, feature, list_segments):
		""" Add to the dataset the data corresponding to the segments specified in input

		input:
		feature: a string corresponding the available features for segments in mvnx:
				'orientation'
				'position'
				'velocity'
				'acceleration'
		list_segments: list of string containing the segments name to retrieve data
		Add all data in the mocap_data list and returns it
		"""
		segments_mvnx = self.mvnx_tree[0].get_list_tags('segments')

		if(feature in segment_features):
			for segment in list_segments:
				if((segment not in segments_mvnx) or # check if the segment name exist and not already in the features list
					((feature + '_' + segment) in self.list_features[0])): 
					continue
				
				id_segment = self.mvnx_tree[0].get_id_segment(segment)

				for i in range(self.n_seq):
					if(feature in self.list_features[0]): # check if the feature is already loaded
						data = self.get_data_by_features(feature, i)
					else:
						data = self.mvnx_tree[i].get_data(feature)

					dimension = int(np.shape(data)[1]/len(segments_mvnx))

					self.mocap_data[i].append(np.asarray(
						data[:,id_segment * dimension : id_segment * dimension + dimension]))

				self.list_features[0].append(feature + '_' + segment)
				self.list_features[1].append(dimension)

		return self.mocap_data


	def add_features_by_joints(self, feature, list_joints):
		""" Add to the dataset the data corresponding to the joints specified in input

		input:
		feature: a string corresponding the available features for joints in mvnx:
				'jointAngle'
				'jointAngleXZY'
				'angularVelocity'
				'angularAcceleration'
		list_joints: list of string containing the joint name to retrieve data
		Add all data in the mocap_data list and returns it
		"""
		joints_mvnx = self.mvnx_tree[0].get_list_tags('joints')
		
		if(feature in joint_features):
			for joint in list_joints:
				if((joint not in joints_mvnx) or # check if the joint name exist and not already in the features list
					((feature + '_' + joint) in self.list_features[0])): 
					continue
				
				id_joint = self.mvnx_tree[0].get_id_joint(joint)

				for i in range(self.n_seq):
					if(feature in self.list_features[0]): # check if the feature is already loaded
						data = self.get_data_by_features(feature, i)
					else:
						data = self.mvnx_tree[i].get_data(feature)

					dimension = int(np.shape(data)[1]/len(joints_mvnx))

					self.mocap_data[i].append(np.asarray(
						data[:,id_joint * dimension : id_joint * dimension + dimension]))

				self.list_features[0].append(feature + '_' + joint)
				self.list_features[1].append(dimension)

		return self.mocap_data


	def add_features_by_sensors(self, feature, list_sensors):
		""" Add to the dataset the data corresponding to the inertial sensors specified in input

		input:
		feature: a string corresponding the available features for sensor in mvnx:
				'sensorOrientation',
				'sensorAngularVelocity',
				'sensorAcceleration'
		list_joints: list of string containing the sensor name to retrieve data
		Add all data in the mocap_data list and returns it
		"""
		sensors_mvnx = self.mvnx_tree[0].get_list_tags('sensors')
		
		if(feature in sensor_features):
			for sensor in list_sensors:
				if((sensor not in sensors_mvnx) or # check if the sensor name exist and not already in the features list
					((feature + '_' + sensor) in self.list_features[0])): 
					continue
				
				id_sensor = self.mvnx_tree[0].get_id_sensor(sensor)

				for i in range(self.n_seq):
					if(feature in self.list_features[0]): # check if the feature is already loaded
						data = self.get_data_by_features(feature, i)
					else:
						data = self.mvnx_tree[i].get_data(feature)

					dimension = int(np.shape(data)[1]/len(sensors_mvnx))

					self.mocap_data[i].append(np.asarray(
						data[:,id_sensor * dimension : id_sensor * dimension + dimension]))

				self.list_features[0].append(feature + '_' + sensor)
				self.list_features[1].append(dimension)

		return self.mocap_data


	def get_data_by_features(self, name_feature, num_sequence):
		"""
		Input: a string for the name of the feature
		Return the data corresponding to the feature in input
		"""
		index_feature =  self.list_features[0].index(name_feature)
		return self.mocap_data[num_sequence][index_feature]

	def get_dimension_features(self, list_features):
		dimension_list = [] 
		for feature in list_features:
			index_feature =  self.list_features[0].index(feature)
			dimension_list.append(self.list_features[1][index_feature])
		return dimension_list
		

	def get_real_labels(self, timestamps):
		""" Return the real labels related to the timestamps in input
		
		"""
		self.real_labels = [[]]

		for i in range(len(timestamps)):
			flag = 0
			time = (np.asarray(timestamps[i]) - timestamps[i][0])



			end = self.ref_data[i][0][2]
			labels = self.ref_data[i][0][0]

			for t in time:
				if(t >= end[flag]):
					flag += 1
					if(flag >= len(end)):
						break
				self.real_labels[i].append(labels[flag])

			if(i < self.n_seq -1):
				self.real_labels.append([])

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


	def add_data_glove(self, info_signal):
		path = self.path_data + self.eglove_folder
		list_files = os.listdir(path)
		list_files.sort()

		list_items = info_signal.findGroup('list').tail().toString().split(' ')

		self.glove_timestamps = [[]]
		self.data_glove = []

		is_norm = info_signal.find('norm').toString()
		if(is_norm == ''):
			is_norm = 0
		else:
			is_norm = int(is_norm)

		for file, i in zip(list_files, range(self.n_seq)):
			with open(path + '/' + list_files[i], 'rt') as f:
				reader = csv.reader(f, delimiter=' ')

				data_forces = []
				data_angles = []

				for row in reader:
					data_forces.append(list(map(float, row[0:4])))
					data_angles.append(list(map(float, row[4:7])))

					self.glove_timestamps[i].append(float(row[7]))


				data_reduce = np.asarray(data_forces)

				if(is_norm):				
					data_reduce = np.linalg.norm(data_forces, axis = 1)
					data_reduce = np.expand_dims(data_reduce, axis=1)

				# data_angles = np.asarray(data_angles)

				# data_reduce = np.concatenate((data_reduce, data_angles), axis = 1)

				self.data_glove.append(data_reduce)
				
			if(i < self.n_seq - 1):
				self.glove_timestamps.append([])

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






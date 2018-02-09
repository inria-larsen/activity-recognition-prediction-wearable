from xsens_parser import mvnx_tree
from anvil_parser import anvil_tree
import numpy as np
import os

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
	eglove_folder = "/eglove/"
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
			T = self.mvnx_tree[i].get_timestamp()
			self.mocap_data[i].append(self.mvnx_tree[i].get_timestamp())
			if(i < self.n_seq - 1):
				self.mocap_data.append([])

		# set the timestamp as the first data on the list of features
		self.list_features[0].append('time')
		self.list_features[1].append(1)
		return path


	def load_labels_ref(self):
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

		self.real_labels = [[]]
		
		for file, i in zip(list_files, range(self.n_seq)):
			ref = anvil_tree(path + file)

			labels, start, end = ref.get_data()
			# for in ref.get_data():
			# 	self.ref_data[i]
			self.ref_data[i].append(ref.get_data())

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
				print(feature, sub_list)
				if(feature == 'centerOfMass'):
					for i in range(self.n_seq):
						com = self.mvnx_tree[i].get_data(feature)
						if(((sub_list[0] == 'all') or (sub_list[0] == ''))):
							self.mocap_data[i].append(com)
							continue
						if('z' in sub_list[0]):
							self.mocap_data[i].append(com[:,2])
						if('y' in sub_list[0]):
							self.mocap_data[i].append(com[:,2])
						if('x' in sub_list[0]):
							self.mocap_data[i].append(com[:,2])
						
					if(((sub_list[0] == 'all') or (sub_list[0] == ''))):
						self.list_features[0].append(feature)
						self.list_features[1].append(3)
		
					if('z' in sub_list[0]):		
						self.list_features[0].append('COM_Z')
						self.list_features[1].append(1)

					if('y' in sub_list[0]):		
						self.list_features[0].append('COM_Y')
						self.list_features[1].append(1)

					if('x' in sub_list[0]):		
						self.list_features[0].append('COM_X')
						self.list_features[1].append(1)

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

			elif(feature == 'CoMVelocity'):
				for i in range(self.n_seq):
					com = self.mvnx_tree[i].get_data('centerOfMass')
					com_vel = np.diff(com, axis=0)
					com_vel = np.insert(com_vel, 0, 0, axis=0)

					self.mocap_data[i].append(com_vel)

				self.list_features[0].append(feature)
				self.list_features[1].append(3)








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

		for i in range(self.n_seq):
			flag = 0
			time = timestamps[i]

			

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

	def get_nbr_sequence(self):
		return self.n_seq

	def get_list_states(self):
		return self.list_states






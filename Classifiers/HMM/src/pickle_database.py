from data_base import DataBase
import numpy as np
import sys
import tools
import pandas as pd 
import os
import csv
import argparse
import configparser

import pickle


if __name__ == '__main__':
	"""
	This script allows to save data from mvnx file into one file
	It uses the information from a .ini file to select features to keep from the file
	"""

	# Get arguments
	parser=argparse.ArgumentParser()
	parser.add_argument('--file', '-f', help='Configuration file', default="config_dataset.ini")
	parser.add_argument('--config', '-c', help='Configuration type', default="DEFAULT")
	args=parser.parse_args()
	config_file = args.file
	config_type = args.config

	local_path = os.path.abspath(os.path.dirname(__file__))

	# Parameters configuration
	config = configparser.ConfigParser()
	config.read('config_file/' + config_file)

	path_data = config[config_type]["path_data"]
	path_labels = config[config_type]["path_labels"]
	path_save = config[config_type]["path_save"]

	list_participant = os.listdir(path_data)
	list_participant.sort()

	print('Loading data...')

	data_win = []
	labels = []
	list_seq = []

	name_track = config[config_type]["name_track"]

	for participant in list_participant:
		path_seq = path_data + participant + '/' 

		print('Loading: ' + participant)
		
		list_files = os.listdir(path_seq)
		list_files.sort()

		for file in list_files:
			name_seq = os.path.splitext(file)[0]
			data_base = DataBase(path_seq, name_seq)
			data_base.load_mvnx_data(path_seq)
			data, time, list_features, dim_features = tools.load_data_from_dataBase(data_base, config)
			file_label = path_labels + participant + '/' + os.path.splitext(name_seq)[0] + '.labels.csv'
			real_labels, list_states = tools.load_labels_ref(time, file_label, name_track, GT = 1)

			time = np.expand_dims(time, axis=1)
			all_data = np.concatenate((time, data), axis=1)

			df = pd.DataFrame(all_data, index=range(len(time)))
			list_features.insert(0, 'timestamp')
			df.columns = list_features

			data_win.append(all_data)
			labels.append(real_labels)
			list_seq.append(os.path.splitext(name_seq)[0])

			# Save database
			tools.save_data_to_dump(path_save, list_seq, data_win, labels, list_states, list_features)







from hmm_model import ModelHMM
import numpy as np
import sys
import visualization_tools as v_tools
import tools
import os
import argparse
import configparser

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
	"""
	Create HMMM models based on parameters in a configuration file in input.

	Exemple:
		python model_simple.py -f config_model.ini -c test_model
	"""
	local_path = os.path.abspath(os.path.dirname(__file__))

	# Get arguments
	parser=argparse.ArgumentParser()
	parser.add_argument('--file', '-f', help='Configuration file', default="config_model.ini")
	parser.add_argument('--config', '-c', help='Configuration type', default="DEFAULT")
	args=parser.parse_args()
	config_file = args.file
	config_type = args.config

	local_path = os.path.abspath(os.path.dirname(__file__))

	# Parameters configuration
	config = configparser.ConfigParser()
	config.read('config_file/' + config_file)

	path_data = config[config_type]["path_data"]
	path_model = config[config_type]["path_model"]
	name_model = config[config_type]["name_model"]
	tracks = config[config_type]["tracks"].split(' ')
	list_features_final = config[config_type]["list_features"].split(' ')
	flag_save = int(config[config_type]["flag_save"])
	ratio_split = list(map(int, config[config_type]["ratio_split_sets"].split(' ')))
	nbr_cross_val = int(config[config_type]["nbr_cross_validation"])

	print('Loading data...')
	data_win, real_labels, list_states, list_features = tools.load_data_from_dump(path_data)

	# Loop on all the tracks from the taxonomy
	for num_track, name_track in enumerate(tracks):
		F1_score = []

		for n_iter in range(nbr_cross_val):
			data_train, labels_train, data_test, labels_test, id_train, id_test = tools.split_data_base(data_win, real_labels[num_track], ratio_split)

			# Keep only the data related to the final list of features
			train_set = tools.reduce_data_to_features(data_train, list_features, list_features_final)
			test_set = tools.reduce_data_to_features(data_test, list_features, list_features_final)
			dim_features = np.ones(len(list_features_final))

			# Training the model
			model = ModelHMM()
			model.train(train_set, labels_train, list_features_final, dim_features)

			# Testing the model
			pred_labels, proba = model.test_model(test_set)

			F1_temp = []
			for i in range(len(labels_test)):
				F1_temp.append(tools.compute_F1_score(labels_test[i], pred_labels[i], list_states[num_track]))

			F1_score.append(np.mean(F1_temp))

		if(flag_save):
			model.save_model(path_model, name_model, "load_handling_" + name_track)
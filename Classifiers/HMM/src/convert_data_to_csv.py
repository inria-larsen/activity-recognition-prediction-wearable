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

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
	"""
	This script allows to convert data from mvnx file into csv file
	It uses the information from a .ini file to select features to keep from the file
	"""

	yarp.Network.init()
	rf = yarp.ResourceFinder()
	rf.setDefaultContext("online_recognition")
	rf.setDefaultConfigFile("convert.ini")
	rf.configure(sys.argv)

	path = rf.find('path_data_base').toString()
	labels_folder = rf.find('labels_folder').toString()

	list_participant = os.listdir(path)
	list_participant.sort()
	print(list_participant)

	print('Loading data...')

	timestamps = []
	data_win = []

	list_participant = ['909']

	# Loop on all participants
	for participant in list_participant:
		path_data = path + '/' + participant + '/mvnx/'
		path_save = path + '/' + participant + '/xsens_csv/'

		print('Loading: ' + participant)
		
		list_files = os.listdir(path_data)
		list_files.sort()

		for file in list_files:
			name_seq = os.path.splitext(file)[0]
			data_base = DataBase(path + '/' + participant, name_seq)
			data_base.load_mvnx_data(path_data)
			data, time, list_features, dim_features = tools.load_data_from_dataBase(data_base, rf)

			list_all_measure = []
			for feature, dim in zip(list_features, dim_features):
				for i in range(dim):
					list_all_measure.append(feature + '_' + str(i))

			list_all_measure.insert(0, 'timestamps_(s)')

			

			time = np.expand_dims(time, axis=1)
			all_data = np.concatenate((time, data), axis=1)

			df = pd.DataFrame(all_data, columns = list_all_measure, index=range(len(time)))

			df.index.name = 'frame'

			if(not(os.path.isdir(path_save))):
				os.makedirs(path_save)
			df.to_csv(path_save + '/' + name_seq + '.xsens.csv')

			list_track = ['general_posture', 'detailed_posture', 'current_action']
			header = ['timestamps']

			count_track = 0
			for name_track in list_track:
				# Load the annotation files
				df_labels = pd.read_csv(path + '/' + participant + '/' + labels_folder + '/' + 'Segments_' + name_seq + '_' + name_track + '_3A.csv')
				t_sample = (np.asarray(time) - time[0]).tolist()
	
				count = 0
				labels = []
				for t in t_sample:
					labels_temp = df_labels.iloc[count, 3:6].values

					# Change the label for shorter codes
					for lab, i in zip(labels_temp, range(len(labels_temp))):
						labels_temp[i] = labels_temp[i].replace("standing", "St")
						labels_temp[i] = labels_temp[i].replace("standing", "St").replace("walking", "Wa")
						labels_temp[i] = labels_temp[i].replace("standing", "St").replace("crouching", "Cr")
						labels_temp[i] = labels_temp[i].replace("standing", "St").replace("kneeling", "Kn")

						labels_temp[i] = labels_temp[i].replace("standing", "St").replace("upright", "U")
						labels_temp[i] = labels_temp[i].replace("standing", "St").replace("_forward","")
						labels_temp[i] = labels_temp[i].replace("standing", "St").replace("strongly_bent","BS")
						labels_temp[i] = labels_temp[i].replace("standing", "St").replace("bent","BF")
						labels_temp[i] = labels_temp[i].replace("standing", "St").replace("overhead_work_hands_above_head","OH")
						labels_temp[i] = labels_temp[i].replace("standing", "St").replace("overhead_work_elbow_at_above_shoulder","OS")

						labels_temp[i] = labels_temp[i].replace("standing", "St").replace("fine_manipulation_no_task","idle")
						labels_temp[i] = labels_temp[i].replace("standing", "St").replace("release_no_task","idle")
						labels_temp[i] = labels_temp[i].replace("standing", "St").replace("reaching_no_task","idle")
						labels_temp[i] = labels_temp[i].replace("standing", "St").replace("picking","Pi")
						labels_temp[i] = labels_temp[i].replace("placing","Pl")
						labels_temp[i] = labels_temp[i].replace("release","Rl")
						labels_temp[i] = labels_temp[i].replace("reaching","Re")
						labels_temp[i] = labels_temp[i].replace("screwing","Sc")
						labels_temp[i] = labels_temp[i].replace("fine_manipulation","Fm")
						labels_temp[i] = labels_temp[i].replace("carrying","Ca")
						labels_temp[i] = labels_temp[i].replace("idle","Id") 

					labels.append(labels_temp)

					if(t > df_labels.iloc[count, 1:2].values):
						count += 1
						if(count == len(df_labels)):
							labels.append(labels_temp)
							break

				t_sample = pd.DataFrame(time)


				for i in range((np.shape(labels)[1])):
					header.append(name_track + '_Annotator' + str(i+1))

				labels = pd.DataFrame(labels)

				if(count_track == 0):	
					df_annotation =  pd.concat([t_sample, labels], axis = 1, join='inner')
					count_track = 1
				else:
					df_annotation = pd.concat([df_annotation, labels], axis = 1, join='inner')


			df_annotation.columns = header
			df_annotation.index.name = 'frame'
			

			path_save = path + '/' + participant + '/labels_csv/'

			if(not(os.path.isdir(path_save))):
				os.makedirs(path_save)
			df_annotation.to_csv(path_save + '/' + name_seq + '.labels.csv')




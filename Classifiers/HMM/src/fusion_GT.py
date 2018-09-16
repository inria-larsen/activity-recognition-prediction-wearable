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
import re

import warnings
warnings.filterwarnings("ignore")




if __name__ == '__main__':
	"""
	This scipt is used to fusion the annotations from the different level of 
	annotation into on file of ground truth
	"""
	yarp.Network.init()
	rf = yarp.ResourceFinder()
	rf.setDefaultContext("online_recognition")
	rf.setDefaultConfigFile("default.ini")
	rf.configure(sys.argv)

	path = rf.find('path_data_base').toString()
	path_model = rf.find('path_model').toString()
	name_model = rf.find('name_model').toString()
	name_track = rf.find('level_taxonomy').toString()
	labels_folder = rf.find('labels_folder').toString()

	list_participant = os.listdir(path)
	list_participant.sort()

	list_participant = ['541', '909', '3327', '5124', '5521', '5535', '8410', '9266', '9875']

	name_track = ['general_posture', 'detailed_posture', 'current_action']

	for participant, nbr in zip(list_participant, range(len(list_participant))):
		path_data = path + '/' + participant + '/labels_3A/'
		
		files_list = [f for f in os.listdir(path_data) if os.path.isfile(os.path.join(path_data, f)) and f.startswith("Participant_" + participant) and f.endswith("_GT.csv")]
		
		files = [f.rsplit("_",3)[0] for f in os.listdir(path_data) if os.path.isfile(os.path.join(path_data, f)) and f.startswith("Participant_") and f.endswith("general_posture_GT.csv")]

		for file in files:
			count = 0
			for track in name_track:
				
				if(count == 0):
					df_labels = pd.read_csv(path_data + file + '_' + track + '_GT.csv')
					df_labels.rename(columns={'labels': track}, inplace=True)
					count = 1
				else:
					df_temp = pd.read_csv(path_data + file + '_' + track + '_GT.csv')
					df_labels = pd.concat([df_labels, df_temp['labels'].rename(track)], axis=1, join='inner')

				if(track == 'detailed_posture'):
					df_temp =  deepcopy(df_labels[track].values)
					for i in range(len(df_temp)):
						df_temp[i] = df_temp[i].replace("standing_", "")
						df_temp[i] = df_temp[i].replace("walking_", "")
						df_temp[i] = df_temp[i].replace("crouching_", "")
						df_temp[i] = df_temp[i].replace("kneeling_", "")

						df_temp[i] = df_temp[i].replace("St_", "")
						df_temp[i] = df_temp[i].replace("Wa_", "")
						df_temp[i] = df_temp[i].replace("Cr_", "")
						df_temp[i] = df_temp[i].replace("Kn_", "")

					df_temp	= pd.DataFrame({'details': df_temp})
					df_labels = pd.concat([df_labels, df_temp], axis=1, join='inner')


			df_labels = df_labels.replace("standing", "St")
			df_labels = df_labels.replace("walking", "Wa")
			df_labels = df_labels.replace("crouching", "Cr")
			df_labels = df_labels.replace("kneeling", "Kn")

			df_labels = df_labels.replace("upright", "U")
			df_labels = df_labels.replace("_forward","")
			df_labels = df_labels.replace("strongly_bent","BS")
			df_labels = df_labels.replace("bent","BF")
			df_labels = df_labels.replace("overhead_work_hands_above_head","OH")
			df_labels = df_labels.replace("overhead_work_elbow_at_above_shoulder","OS")

			df_labels = df_labels.replace("fine_manipulation_no_task","idle")
			df_labels = df_labels.replace("Rl_no_task","idle")
			df_labels = df_labels.replace("reaching_no_task","idle")
			df_labels = df_labels.replace("picking","Pi")
			df_labels = df_labels.replace("placing","Pl")
			df_labels = df_labels.replace("release","Rl")
			df_labels = df_labels.replace("reaching","Re")
			df_labels = df_labels.replace("screwing","Sc")
			df_labels = df_labels.replace("fine_manipulation","Fm")
			df_labels = df_labels.replace("carrying","Ca")
			df_labels = df_labels.replace("idle","Id")      

			df_labels.to_csv(path_data + file + '_GT.csv', index = False)


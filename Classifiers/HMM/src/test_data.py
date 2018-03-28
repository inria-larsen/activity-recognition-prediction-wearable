import numpy as np
import pandas as pd 
import csv
from copy import deepcopy
import seaborn as sb
import matplotlib.pyplot as plt

list_participant = [909, 541, 5521, 3327]

ListActivities = [
    'walking_upright',
    'standing_upright',
    'standing_bent_forward',
    'standing_strongly_bent_forward',
    'standing_overhead_work_hands_above_head',
    'kneeling_upright',
    'kneeling_bent'
    ]


dataFrame_stand = pd.read_csv('standing_upright.csv')

# for state in ListActivities:
if(1):
	state = ListActivities[-1]
	dataFrame = pd.read_csv(state + '.csv')

	all_data = dataFrame['com_z']

	data_normalize1 = []
	data_normalize2 = []

	list_end = np.int_(np.zeros(len(list_participant) + 1))

	for subject, nb_sujet in zip(list_participant, range(len(list_participant))):

		df_stand = dataFrame_stand[dataFrame_stand.id == subject]

		data_init1 = df_stand['com_z'].median()
		data_init2 = df_stand['com_z'].median()


		df = dataFrame[dataFrame.id == subject]

		list_end[nb_sujet+1:] += int(len(df))

		for data in df['com_z']:
			data_normalize1.append(data - data_init1)
			data_normalize2.append(data/data_init2 - 1)



	color = ['red', 'blue', 'yellow', 'green']




for i in range(len(list_participant)):
	plt.boxplot([data_normalize1, data_normalize2])


plt.show()
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

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
	df_states = pd.read_csv('states.csv')
	df_data = pd.read_csv('save_features.csv')

	frames = [df_states, df_data]

	df_total = pd.concat(frames, axis = 1, join = 'inner')

	n_states = np.max(df_states) + 1

	



	# f_score = tools.fisher_score2(data, obs)
	# list_id_sort = np.argsort(f_score)

	# for id_sort in list_id_sort:
	# 	print(list_all_features[id_sort], f_score[id_sort])








	# for nbr_test in range(nbr_cross_val):
	# 	confusion_matrix = np.zeros((len(list_states), len(list_states)))

	# 	# data_ref, labels_ref, data_test, labels_test, data_val, labels_val, id_train, id_test, id_val = tools.split_data_base(data_win, real_labels, ratio)
	# 	data_ref, labels_ref, data_test, labels_test, id_train, id_test = tools.split_data_base2(data_win, real_labels, ratio)





	# 			model = ModelHMM()
	# 			model.train(data_ref, labels_ref, sub_list_features, sub_dim_features)

	# 			if(save):
	# 				model.save_model(path_model, name_model, "load_handling")


	# 			#### Test

	# 			# ref_labels_detailed = []

	# 			time_test = []
	# 			for id_subject in id_test:
	# 				time_test.append(timestamps[id_subject])
	# 				# ref_labels_detailed.append(real_labels_detailed[id_subject])


	# 			predict_labels, proba = model.test_model(data_test)

	# 			for i in range(len(predict_labels)):
	# 				conf_mat = tools.compute_confusion_matrix(predict_labels[i], labels_test[i], list_states)
	# 				confusion_matrix += conf_mat

	# 			prec_total, recall_total, F1_score = tools.compute_score(confusion_matrix)
	# 			acc = tools.get_accuracy(confusion_matrix)
	# 			# print(confusion_matrix)

	# 			F1_S += F1_score/nbr_cross_val



	# 		if(len(score) == 0):
	# 			score.append(F1_S)
	# 			best_features.append(sub_list_features)
	# 			dim_best.append(sub_dim_features)
	# 		else:
	# 			for num in range(len(score)):
	# 				if(F1_S > score[num]):
	# 					score.insert(num, F1_S)
	# 					best_features.insert(num, sub_list_features)
	# 					dim_best.insert(num, sub_dim_features)
	# 					break

	# 				if(num == len(score)-1):
	# 					score.append(F1_S)
	# 					best_features.append(sub_list_features)
	# 					dim_best.append(sub_dim_features)


	# score_totaux = pd.DataFrame(
	# 	{'best_features': best_features,
	# 	 'score': score,
	# 	})
	# score_totaux.to_csv('results' + ".csv", index=False)

	plt.show()



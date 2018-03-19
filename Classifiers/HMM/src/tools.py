# from features_organizer import FeaturesOrganizer
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split


def mean_and_cov(all_data, labels, n_states, n_feature):
	""" Compute the means and covariance matrix for each state 

	Return a vector of scalar and a vector of matrix corresponding to the mean 
	of the distribution and the covariance
	"""

	data = [[]]
	for i in range(n_states - 1):
		data.append([])

	for i in range(len(labels)):
		num_state = labels[i]
		if(len(data[num_state])<=0):
			data[num_state].append(all_data[i])
		else:
			data[num_state] = np.vstack((data[num_state], all_data[i]))

	sigma = np.zeros(((n_states,n_feature,n_feature)))
	mu = np.zeros((n_states, np.sum(n_feature)))

	for i in range(n_states):
		mu[i] = np.mean(data[i], axis=0)
		sigma[i] = np.cov(data[i].T)

	return mu, sigma

def quaternion_to_rotation(q, trans):
	Rot = np.zeros((4, 4))

	Rot[0,0] = 1 - 2*q[2]*q[2] - 2*q[3]*q[3]
	Rot[0,1] = 2*q[1]*q[2] - 2*q[0]*q[3]
	Rot[0,2] = 2*q[1]*q[2] - 2*q[0]*q[3]
	Rot[0,3] = -trans[0]

	Rot[1,0] = 2*q[1]*q[2] + 2*q[0]*q[3]
	Rot[1,1] = 1 - 2*q[1]*q[1] - 2*q[3]*q[3]
	Rot[1,2] = 2*q[2]*q[3] - 2*q[0]*q[1]
	Rot[1,2] = -trans[1]

	Rot[2,0] = 2*q[1]*q[3] - 2*q[0]*q[2]
	Rot[2,1] = 2*q[2]*q[3] + 2*q[0]*q[1]
	Rot[2,2] = 1 - 2*q[1]*q[1] - 2*q[2]*q[2]
	Rot[2,3] = -trans[2]

	Rot[2,3] = 0
	Rot[2,3] = 0
	Rot[2,3] = 0
	Rot[2,3] = 1

	return Rot

def normalize_position(pose_data, quaternion):
	origin = pose_data[0, 0:3]
	origin_orientation = quaternion[0, 0:4]

	position = np.zeros((np.shape(pose_data)))

	for i in range(len(pose_data)):
		for j in range(0, 23):
			if(j == 0):
				Rot_root = quaternion_to_rotation(origin_orientation, origin)
			else:
				Rot_root = quaternion_to_rotation(quaternion[i, j*4:j*4+4], pose_data[i, j*3:3])

			# position[i, j : j+3] = Rot_root.dot(pose_data[i, j : j+3]) + origin
			pose_in = np.append(pose_data[i, j : j+3], [0])
			pose_out = Rot_root.dot(pose_in)
			position[i, j : j+3] = pose_out[0:3]

	return position


def split_data_base(data_set, labels, ratio):
	"""
	This function allows to split a database into three subset:
	- Reference subset
	- Validation subset
	- Test subset
	The number of elements in each subset correspond to the ratio in input such as ratio is a list of float or int such as sum(ratio) = 100
	"""

	nbr_sequences = len(data_set)

	base_ref = []
	labels_ref = []
	base_test = []
	labels_test = []
	base_val = []
	labels_val = []

	id_train, id_subset = train_test_split(np.arange(nbr_sequences), train_size=ratio[0]/100)
	id_test, id_val = train_test_split(id_subset, train_size=(ratio[2]*100/(100-ratio[0]))/100)

	for i in id_train:
		base_ref.append(data_set[i])
		labels_ref.append(labels[i])

	for i in id_test:
		base_test.append(data_set[i])
		labels_test.append(labels[i])

	for i in id_val:
		base_val.append(data_set[i])
		labels_val.append(labels[i])
	
	return base_ref, labels_ref, base_test, labels_test, base_val, labels_val, id_train, id_test, id_val



def split_data_base2(data_set, labels, ratio):
	"""
	This function allows to split a database into two subsets:
	- Reference subset
	- Test subset
	The number of elements in each subset correspond to the ratio in input such as ratio is a list of float or int such as sum(ratio) = 100
	"""

	nbr_sequences = len(data_set)

	base_ref = []
	labels_ref = []
	base_test = []
	labels_test = []

	id_train, id_test = train_test_split(np.arange(nbr_sequences), train_size=ratio[0]/100)

	for i in id_train:
		base_ref.append(data_set[i])
		labels_ref.append(labels[i])

	for i in id_test:
		base_test.append(data_set[i])
		labels_test.append(labels[i])
	
	return base_ref, labels_ref, base_test, labels_test, id_train, id_test

# def synchronize_glove(data_glove, time_glove):
# 	return data_glove_sync, time_glove_sync, data


def generate_list_features(n_features, possible_features):
	list_features = []
	dim_features = []
	for i in range(n_features):
		id_feature = int(np.floor(np.random.rand() * (len(possible_features[0]))))

		while((possible_features[0][id_feature] in list_features) or 
			((possible_features[0][id_feature] == 'CoMZ') and ('centerOfMass' in list_features)) or
			((possible_features[0][id_feature] == 'centerOfMass') and ('CoMZ' in list_features))):

			id_feature = int(np.floor(np.random.rand() * (len(possible_features[0]))))
		list_features.append(possible_features[0][id_feature])
		

	for id_feature in range(len(list_features)):
		index = possible_features[0].index(list_features[id_feature])
		dim_features.append(possible_features[1][index])

	return list_features, dim_features

# def update_list_name_features(list_features, list_all_features, list_name_features):
# 	name_features = []
# 	for features in list_features:
# 		index = list_all_features.index(features)
# 		name_features.append(list_name_features[index])
# 	return name_features


# def slide_windows(data, size_window):
# 	win = int(size_window/2)
# 	data_out = []

# 	for i in np.arange(win, len(data)-win, win):
# 		average = np.mean(data[i-win : i+win+1], axis=0)
# 		data_out.append(average)

# 	data_windows = np.zeros((np.shape(data_out)))
# 	for i in range(len(data_out)):
# 		data_windows[i,:] = data_out[i]
# 	return data_windows


# def all_data_concatenate(list_features, all_data):
# 	data_conc = []
# 	for features in list_features:
# 		data = all_data.get_data_by_features(features)
# 		if(len(data_conc)<=0):
# 			data_conc = data
# 		else:
# 			data_conc = np.hstack((data_conc, data))
# 	return data_conc

# def set_name_ouput(list_features, all_features, name_features):
# 	features_sorted = sorted(all_features)
# 	id_feature = all_features.index(list_features[0])
# 	name_output_file = name_features[id_feature]
# 	name_output = []
# 	name_output.append(name_features[id_feature])
# 	for i in range(1, len(list_features)):
# 		id_feature = all_features.index(list_features[i])
# 		name_output_file += '_' + name_features[id_feature]
# 		name_output.append(name_features[id_feature])
# 	name_output = sorted(name_output)
# 	name_output_file = name_output[0]
# 	for i in range(1, len(name_output)):
# 		name_output_file += '_' + name_output[i]
# 	return name_output_file

def compute_confusion_matrix(pred_labels, real_labels, list_states):
	n_states = len(list_states)
	confusion_matrix = np.zeros((n_states, n_states)).astype(int)
	for j in range(len(real_labels)):
		index_real = list_states.index(real_labels[j])
		index_pred = list_states.index(pred_labels[j])
		confusion_matrix[index_real, index_pred] += 1
	return confusion_matrix.astype(int)

def prepare_segment_analysis(timestamps, prediction_labels, reference_labels):
	ground_truth = [[]]
	prediction = [[]]
	time = [[]]
	id_sample_start = [[]]
	id_sample_end = [[]]

	for i in range(len(prediction_labels)):
		for t in range(len(prediction_labels[i])):
			if(t == 0):
				prediction[i].append(prediction_labels[i][t])
				time[i].append(timestamps[i][t])
				ground_truth[i].append(reference_labels[i][t])
				id_sample_start[i].append(t)

			else:
				if((prediction_labels[i][t] != prediction_labels[i][t-1]) or (reference_labels[i][t] != reference_labels[i][t-1])):
					prediction[i].append(prediction_labels[i][t])
					ground_truth[i].append(reference_labels[i][t])
					time[i].append(timestamps[i][t])
					id_sample_start[i].append(t)
					id_sample_end[i].append(t-1)
					continue

		if(len(ground_truth) < len(prediction_labels)):
			ground_truth.append([])
			prediction.append([])
			time.append([])
			id_sample_start.append([])
			id_sample_end[i].append(t)
			id_sample_end.append([])

	return time, ground_truth, prediction, id_sample_start, id_sample_end




# def normalize_data(data, xmin, xmax):
# 	data_norm = 2*(data - xmin)/(xmax - xmin) - 1
# 	return data_norm

# # def compute_results(model, path):
# # 	data_test = model.get_data_test()
# # 	print('size', np.shape(data_test))
# # 	index = model.get_test_set_index()
# # 	print('index', index)
# # 	labels = model.get_ref_labels(index[0])
# # 	print(len(labels))
# # 	list_states = model.get_list_states()

# # 	obs = data_test[0][0]
# # 	Z = model.predict(obs)

# # 	pred_labels = []
# # 	for j in range(len(Z)):
# # 		pred_labels.append(list_states[Z[j]])

# # 	list_states = model.get_list_states()

# # 	confusion_matrix = compute_confusion_matrix(pred_labels, labels, list_states, path)
# # 	return confusion_matrix



# # save confusion matrix in csv file
# def save_confusion_matrix(name_file, path, confusion_matrix, labels):
# 	c = csv.writer(open(path + name_file + '_confusion.csv', "w", newline=''))
# 	line = []
# 	for i in range(len(labels)):
# 		line.append(labels[i].title())
# 	c.writerow(line)
# 	for i in range(len(labels)):
# 		line = [labels[i].title()]
# 		for j in range(len(labels)):
# 			line.append(confusion_matrix[i, j])
# 		c.writerow(line)
# 	return

def plot_confusion_matrix2(path, name, confusion_matrix, list_states, save=0):
	real_len, pred_len = np.shape(confusion_matrix)

	precision_map = np.zeros((real_len, pred_len))
	recall_map = np.zeros((real_len, pred_len))
	for i in range(real_len):
		for j in range(pred_len):
			precision_map[i, j] = confusion_matrix[i, j] / np.sum(confusion_matrix[:,j])
			recall_map[i, j] = confusion_matrix[i, j] / np.sum(confusion_matrix[i])


	precision_map = np.around(precision_map, 4)*100
	recall_map = np.around(recall_map, 4)*100

	precision_pd = pd.DataFrame(data = precision_map, 
				index = list_states
				, columns = list_states
				)


	recall_pd = pd.DataFrame(data = recall_map, 
			index = list_states
			,columns = list_states
			)

	# recall_total, prec_total, F1_score = np.around(compute_score(confusion_matrix), 4)*100

	fig = plt.figure()
	sns.set(font_scale = 1.5)
	# ax1 = plt.subplot("121")
	sns.heatmap(recall_pd, vmax=100., annot=True, fmt='g', cbar=False)
	# ax1.set_title('Recall (%) map - Total = ' + str(recall_total))
	plt.yticks(rotation=0) 
	plt.xticks(rotation=45)
	plt.ylabel('Real labels')
	plt.xlabel('Predict labels')

	plt.title('Recall')

	if(save):
		plt.savefig(name + '_recall.pdf', bbox_inches='tight')

	fig = plt.figure()

	# ax2 = plt.subplot("122")
	sns.heatmap(precision_pd, vmax=100., annot=True, label='big', fmt='g', cbar=False)
	# ax2.set_title('Precision (%) map - Total = ' + str(prec_total))
	plt.yticks(rotation=0)
	plt.xticks(rotation=45)
	plt.ylabel('Real labels')
	plt.xlabel('Predict labels')

	plt.title('Precision')

	if(save):
		plt.savefig(name + '_precision.pdf', bbox_inches='tight')

	return


def plot_confusion_matrix(path, name, title, confusion_matrix, save=0):
	real_len, pred_len = np.shape(confusion_matrix)

	fig = plt.figure(figsize=(12,10))
	sns.set(font_scale = 2.5)
	plt.title(title, fontsize=35)
	# ax1 = plt.subplot("121")
	sns.heatmap(confusion_matrix, annot=True, fmt='g', cbar = False)
	# ax1.set_title('Recall (%) map - Total = ' + str(recall_total))
	plt.yticks(rotation=0) 
	plt.xticks(rotation=45)
	plt.ylabel('Real labels', fontsize=30)
	plt.xlabel('Predicted labels', fontsize=30)

	if(save):
		plt.savefig(path + name + '.pdf', bbox_inches='tight')
	return

# def plot_confusion_matrix_df(confusion_matrix):
# 	print('aaa\n', confusion_matrix)
# 	real_len, pred_len = np.shape(confusion_matrix)

# 	fig = plt.figure()
# 	sns.set(font_scale = 1.5)
# 	# ax1 = plt.subplot("121")
# 	sns.heatmap(confusion_matrix, annot=True, fmt='g', cbar = False)
# 	# ax1.set_title('Recall (%) map - Total = ' + str(recall_total))
# 	plt.yticks(rotation=0) 
# 	plt.xticks(rotation=45)
# 	plt.ylabel('Real labels')
# 	plt.xlabel('Predict labels')

# 	plt.title('Confusion Matrix')

# 	return fig

# def save_tab_score(name_file, path, features, scores):
# 	scores = np.around(np.asarray(scores)*100, decimals = 2)
# 	c = csv.writer(open(path + name_file + '_performance.csv', "w", newline=''))
# 	c.writerow(['Features', 'Precision', 'Recall', 'F1-Score'])
# 	for i in range(len(scores)):
# 		line = [features[i]]
# 		for j in range(len(scores[i])):
# 			line.append(scores[i][j])
# 		c.writerow(line)
# 	return

# def read_confusion_file(path_file):
# 	confusion_matrix = np.zeros((1, 1))
# 	cr = csv.reader(open(path_file, "r"))
# 	# cr = csv.reader(open(path_model + name + '/' + name_csv_file,"r"))
# 	i = -1
# 	for row in cr:
# 		if(i == -1):
# 			labels = row
# 			confusion_matrix = np.zeros((len(labels), len(labels)))
# 			i += 1
# 		else:
# 			if(row == []):
# 				continue
# 			confusion_matrix[i] = row[1:]
# 			i += 1
# 	return confusion_matrix, labels


def compute_score(confusion_matrix):
	n_states = len(confusion_matrix)
	precision = np.zeros((n_states, 1))
	recall = np.zeros((n_states, 1))

	for j in range(n_states):
		TP = confusion_matrix[j,j]
		recall[j] = TP/np.sum(confusion_matrix[j])
		precision[j] = TP/np.sum(confusion_matrix[:,j])
		if(np.isnan(recall[j])):
			recall[j] = 0
		if(np.isnan(precision[j])):
			precision[j] = 0
		
	prec_total = np.sum(precision)/n_states
	recall_total = np.sum(recall)/n_states
	F1_score = 2 * (prec_total * recall_total)/(prec_total + recall_total)

	return prec_total, recall_total, F1_score

def compute_score_by_states(confusion_matrix):
	n_states = len(confusion_matrix)
	precision = np.zeros((n_states))
	recall = np.zeros((n_states))
	F1_score = np.zeros((n_states))

	for j in range(n_states):
		TP = confusion_matrix[j,j]
		recall[j] = TP/np.sum(confusion_matrix[j])
		precision[j] = TP/np.sum(confusion_matrix[:,j])
		if(np.isnan(recall[j])):
			recall[j] = 0
		if(np.isnan(precision[j])):
			precision[j] = 0

		F1_score[j] = 2 * (precision[j] * recall[j])/(precision[j] + recall[j])

	return precision, recall, F1_score


def get_accuracy(confusion_matrix):
	n_states = len(confusion_matrix)
	accuracy = np.sum(np.diag(confusion_matrix))/np.sum(confusion_matrix)
	return accuracy



# def update_best_features(best_features, best_scores, best_n_festures, scores, features_name
# 						, nb_features, n_best, best_path, path, size_window, win):
# 	if(best_scores[0][2] == 0):
# 		best_scores[0] = scores
# 		best_features.append(features_name)
# 		best_path.append(path)
# 		best_n_festures[0] = int(nb_features)
# 		win.append(size_window)
# 	else:
# 		for i in range(len(best_scores)):
# 			if(scores[2] > best_scores[i][2]):
# 				best_scores.insert(i, scores)
# 				best_features.insert(i, features_name)
# 				best_path.insert(i, path)
# 				best_n_festures.insert(i, int(nb_features))
# 				win.insert(i, size_window)
# 				if(len(best_scores) > n_best):
# 					del best_scores[-1]
# 					del best_features[-1]
# 					del best_n_festures[-1]
# 					del best_path[-1]
# 					del win[-1]
# 				break
				
# 			elif(i == (len(best_scores)-1)):
# 				best_scores.append(scores)
# 				best_features.append(features_name)
# 				best_n_festures.append(int(nb_features))
# 				best_path.append(path)
# 				win.append(size_window)
# 				if(len(best_scores) > n_best):
# 					del best_scores[-1]
# 					del best_features[-1]
# 					del best_n_festures[-1]
# 					del win[-1]
# 				break	
# 	return


# def csv_to_latex(csv_file):
# 	latex_tab = pd.read_csv(csv_file, sep=',', index_col = 0)
# 	return latex_tab.to_latex()


# def plot_labels(label_sequence, list_labels):

# 	return

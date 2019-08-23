import numpy as np
import matplotlib.pyplot as plt
import pickle
import csv
import pandas as pd
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn import metrics
import data_processing as pr
from data_base import DataBase
import matplotlib
from copy import deepcopy
import random 
import pickle
import os

def mean_and_cov(all_data, labels, n_states, list_features):
	""" Compute the means and covariance matrix for each state 

	Return a vector of scalar and a vector of matrix corresponding to the mean 
	of the distribution and the covariance
	"""
	n_feature = len(list_features)
	df_data = pd.DataFrame(all_data, columns = list_features)

	df_labels = pd.DataFrame(labels)
	df_labels.columns = ['state']

	df_total = pd.concat([df_data, df_labels], axis=1)

	data = []
	for state, df in df_total.groupby('state'):
		data.append(df[list_features].values)

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

	if(ratio[2] > 0):

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

	else:
		id_train, id_test = train_test_split(np.arange(nbr_sequences), train_size=ratio[0]/100)

		for i in id_train:
			base_ref.append(data_set[i])
			labels_ref.append(labels[i])

		for i in id_test:
			base_test.append(data_set[i])
			labels_test.append(labels[i])
		
		return base_ref, labels_ref, base_test, labels_test, id_train, id_test




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


def fisher_score(data_c, data_all):
	"""
	Compute the Fisher score for each features of the dataset
	"""
	nbr_features = len(data_all.T)
	n_states = len(data_c)

	fisher_score = np.zeros(nbr_features)

	for i in range(nbr_features):
		mean_i = np.mean(data_all[:,i])
		num = 0
		den = 0
		for j in range(n_states):
			if(len(data_c[j]) == 0):
				continue
			mean_j = np.mean(data_c[j][:,i])
			std_j = np.std(data_c[j][:,i])
			n_j = len(data_c[j])

			num += n_j*(mean_j - mean_i)*(mean_j - mean_i)
			den += n_j*std_j*std_j

		fisher_score[i] = num/den
		if(np.isnan(fisher_score[i])):
			fisher_score[i] = 0

	return fisher_score


def compute_confusion_matrix(pred_labels, real_labels, list_states):
	"""
	Return the confusion matrix (size: n_states*n_states)
	Take in input the prediction label, real label and list states
	"""
	n_states = len(list_states)
	confusion_matrix = np.zeros((n_states, n_states)).astype(int)
	for j in range(len(real_labels)):
		index_real = list_states.index(real_labels[j])
		index_pred = list_states.index(pred_labels[j])
		confusion_matrix[index_real, index_pred] += 1
	return confusion_matrix.astype(int)

def prepare_segment_analysis(timestamps, prediction_labels, reference_labels, id_test):
	ground_truth = [[]]
	prediction = [[]]
	time = [[]]
	id_sample_start = [[]]
	id_sample_end = [[]]

	for i in range(len(prediction_labels)):
		print(np.shape(prediction_labels[i]), np.shape(reference_labels[i]), id_test[i])
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



def plot_confusion_matrix2(path, name, confusion_matrix, list_states, all_in_one = 0, save=0):
	"""
	Plot and save the confusion matrix for prediction and recall score
	"""

	real_len, pred_len = np.shape(confusion_matrix)

	precision_map = np.zeros((real_len, pred_len))
	recall_map = np.zeros((real_len, pred_len))
	for i in range(real_len):
		for j in range(pred_len):
			precision_map[i, j] = confusion_matrix[i, j] / np.sum(confusion_matrix[:,j])
			recall_map[i, j] = confusion_matrix[i, j] / np.sum(confusion_matrix[i])


	precision_map = np.around(precision_map, 2)*100
	recall_map = np.around(recall_map, 2)*100

	precision_pd = pd.DataFrame(data = precision_map, 
				index = list_states
				, columns = list_states
				)

	recall_pd = pd.DataFrame(data = recall_map, 
			index = list_states
			,columns = list_states
			)

	# Put both plots in one figure
	if(all_in_one):
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,8))

		sns.set(font_scale = 2.0)
		sns.heatmap(recall_pd, vmax=100., annot=True, fmt='g', cbar=False, ax=ax1)
		ax1.set_yticklabels(list_states, rotation=0, fontsize = 26) 
		ax1.set_xticklabels(list_states, rotation=45, fontsize = 26)
		ax1.set_ylabel('Real labels', fontsize = 30)
		ax1.set_xlabel('Predicted labels', fontsize = 30)
		ax1.set_title('Recall', fontsize = 30)

		sns.heatmap(precision_pd, vmax=100., annot=True, label='big', fmt='g', cbar=False, ax=ax2)
		ax2.set_yticklabels(list_states, rotation=0, fontsize = 26)
		ax2.set_xticklabels(list_states, rotation=45, fontsize = 26)
		ax2.set_ylabel('Real labels', fontsize = 30)
		ax2.set_xlabel('Predicted labels', fontsize = 30)
		ax2.set_title('Precision', fontsize = 30)

		fig.tight_layout()

	else:
		fig = plt.figure()

		sns.set(font_scale = 1.5)
		sns.heatmap(recall_pd, vmax=100., annot=True, fmt='g', cbar=False)
		plt.yticks(rotation=0) 
		plt.xticks(rotation=45)
		plt.ylabel('Real labels')
		plt.xlabel(' labels')

		plt.title('Recall')
	

	if(save):
		plt.savefig(path + name + '_confusion.pdf', bbox_inches='tight')

	if(all_in_one == 0):
		fig = plt.figure()

	return

def compute_score(confusion_matrix):
	"""
	Return the precision, recall and F1-score based on the confusion matrix
	"""
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

def compute_MCC_score(y_true, y_pred, list_states):
	"""
	Compute the MCC score based on the sequence of real labels, predicted labels and list states
	"""
	index_real = []
	index_pred = []
	for j in range(len(y_true)):
		index_real.append(list_states.index(y_true[j]))
		index_pred.append(list_states.index(y_pred[j]))
	return metrics.matthews_corrcoef(index_real, index_pred)

def compute_F1_score(y_true, y_pred, list_states):
	"""
	Compute the F1-score based on the sequence of real labels, predicted labels and list states
	"""
	return metrics.f1_score(y_true, y_pred, list_states, average = 'micro')

def save_results_to_csv(y_true, y_pred, time, name_file):
	time_ = (time - time[0]).values

	length_seg = []
	start_seg = []
	end_seg = []
	labels_GT = []
	labels_pred = []

	labels_GT.append(y_true[0])
	labels_pred.append(y_pred[0])
	start_seg.append(time_[0])

	index_seg = 0

	length = len(y_true)
	for t in range(1, length):
		if(y_true[t] != labels_GT[index_seg] or y_pred[t] != labels_pred[index_seg]):
			labels_GT.append(y_true[t])
			labels_pred.append(y_pred[t])
			end_seg.append(time_[t])
			start_seg.append(time_[t])
			length_seg.append(end_seg[index_seg] - start_seg[index_seg])
			index_seg += 1

	end_seg.append(time_[-1])
	length_seg.append(end_seg[index_seg] - start_seg[index_seg])


	columns = ['start', 'end', 'length', 'labels_GT', 'labels_pred']

	df_results = pd.DataFrame(
			{'start': start_seg,
			 'end': end_seg,
			 'length': length_seg,
			 'labels_GT' : labels_GT,
			 'labels_pred' : labels_pred,
			})

	df_results = df_results[['start', 'end', 'length', 'labels_GT', 'labels_pred']]

	df_results.to_csv(name_file + ".csv", index=False)
	return


def compute_score_by_states(confusion_matrix):
	"""
	Compute the recognition scores of each state based on the confusion matrix
	"""
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


def load_data_from_dataBase(data_base, config):
	"""
	Load motion capture data from mvnx file
	"""
	timestamps = []
	data_win = []

	window_size = float(config["DEFAULT"]["slidding_window_size"])
	signals = config["DEFAULT"]["list_signals"].split(',')

	nb_port = int(len(signals))
	nb_active_port = 0
	list_features, dim_features = data_base.add_signals_to_dataBase(config)
	glove_on = 0

	if('eglove' in signals):
		info_signal = rf.findGroup('eglove')
		glove_on = int(info_signal.find('enable').toString())
		if(glove_on):
			data_glove, time_glove = data_base.add_data_glove(info_signal)

	sub_data = pr.concatenate_data(data_base, list_features)
	t = []

	t = data_base.get_data_by_features('time')
	t_mocap = t.tolist()
	mocap_data = sub_data


	if(glove_on):
		t_glove = time_glove
		glove_data = data_glove.tolist()

		t_ = 0
		while(t_glove[t_] < t_mocap[0]):
			t_ += 1
		t_start = t_


		while(t_mocap[t_] < t_glove[-1]):
			t_ += 1
			if(t_ == len(t_mocap)):
				break

		t_end = t_+1

		del glove_data[0:t_start]
		del t_glove[0:t_start]

		del glove_data[t_end:]
		del t_glove[t_end:]

		data_force = np.zeros((len(t_mocap), np.shape(glove_data)[1]))

		count = 0

		for k in range(len(t_mocap)):
			data_force[k] = glove_data[count]
			if(t_glove[count] < t_mocap[k]):
				count += 1
				if(count == len(glove_data)):
					break

		data_out, timestamps_out = pr.slidding_window(data_force, t_mocap, window_size)
		data_glove = data_out

	if(window_size > 0):
		data_out, timestamps_out = pr.slidding_window(mocap_data, t_mocap, window_size)
	else:
		data_out = mocap_data
		timestamps_out = t_mocap

	data_win = data_out

	if(glove_on):		
		data_win = np.concatenate((data_win, data_glove) , axis = 1)


	t = timestamps_out
	timestamps = timestamps_out

	if(glove_on):
		if('gloveForces' not in list_features):
			list_features.append('gloveForces')
			dim_features.append(7)

	return data_win, timestamps, list_features, dim_features



def load_data(path, participant, name_seq, name_track, labels_folder, list_features = []):
	"""
	Load data from csv file
	"""
	list_states = []

	data_base = pd.read_csv(path + 'xsens/allFeatures_csv/' + participant + '/' + name_seq + '.csv')
	ref_data = DataBase(path + '/', name_seq)

	list_all_features = list(data_base.columns.values)
	del list_all_features[0:2]
	dim_features = np.ones(len(list_all_features))

	time = data_base['timestamp']

	if(len(list_features) == 0):
		list_features = list_all_features

	labels, states = ref_data.load_labels_ref3A(time, name_track, participant, 1)


	data = data_base[list_features].as_matrix()

	for state in states:
		if(state not in list_states):
			list_states.append(state)
			list_states = sorted(list_states)


	return data, labels, time, list_states, list_features
	# return data, time, list_features

def feature_selection(data, real_labels, list_features):
	obs = []
	obs = data[0]
	lengths = []
	lengths.append(len(data[0]))
	labels = real_labels[0]

	for i in range(1, len(data)):
		obs = np.concatenate([obs, data[i]])
		lengths.append(len(data[i]))
		labels = np.concatenate([labels, real_labels[i]])


	list_states, labels = np.unique(labels, return_inverse=True)
	n_states = len(list_states)

	df_data = pd.DataFrame(obs, columns = list_features)

	df_labels = pd.DataFrame(labels)
	df_labels.columns = ['state']

	df_total = pd.concat([df_data, df_labels], axis=1)

	data = []
	for state, df in df_total.groupby('state'):
		data.append(df[list_features].values)

	f_score = fisher_score(data, obs)
	list_id_sort = np.argsort(f_score).tolist()
	sorted_score = []
	sorted_features = []

	for id_sort in reversed(list_id_sort):
		sorted_features.append(list_features[id_sort])
		sorted_score.append(f_score[id_sort])

	return sorted_features, sorted_score


def load_labels_ref(timestamps, file_labels, name_track, GT = 0):
	""" Load the reference labels from csv file from 3 annotators

	This function updates the reference data and the list of states.
	Take in input the timestamps corresponding to the motion capture data
	Take in input the name of the track used for the labels and the label folder
	Possibility to extract the Ground Truth if GT = 1 (default = 0)

	list_states: sorted list of string containing the list of states
	"""

	timestamps = np.asarray(timestamps)
	time_start = [[]]

	ref_data = []
	real_labels = []
	list_states = []

	# df_labels = pd.read_csv(path + 'Segments_' + self.name_seq + '_' + name_track + '_3A.csv')

	columns_labels = ['t_video']
	for i in range(3):
		columns_labels.append(name_track + '_Annotator' + str(i+1))


	df_labels = pd.read_csv(file_labels)
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

	T = 0.25

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

			real_labels.append((cnt.most_common(1)[0][0]))

	else:
		real_labels = df_labels

	seq_none = [[]]
	count_none = 0
	for i in range(0, len(real_labels)):
		if(GT):
			if(real_labels[i]=='NONE'):
				seq_none[count_none].append(i)
				if(real_labels[i+1] != 'NONE'):
					count_none += 1
					seq_none.append([])

	del seq_none[-1]

	if(GT):
		for i in range(len(seq_none)):
			A = deepcopy(seq_none[i])
			while(len(seq_none[i]) >= 1):
				if(len(seq_none[i]) == 1):
					if(random.randint(0,1)):
						real_labels[seq_none[i][0]] = real_labels[seq_none[i][0]+1]
					else:
						real_labels[seq_none[i][0]] = real_labels[seq_none[i][0]-1]
					del seq_none[i][0]

				else:
					real_labels[seq_none[i][0]] = real_labels[seq_none[i][0]-1]
					real_labels[seq_none[i][-1]] = real_labels[seq_none[i][-1]+1]
					del seq_none[i][0]
					del seq_none[i][-1]

	list_states, l = np.unique(real_labels, return_inverse=True)

	return real_labels, list_states

def get_best_features(file_name):
	best_features = []
	df = pd.read_csv(file_name)
	for i in range(len(df)):
		line = df['best_features'].values[i] # Find the set of features with the best score
		line = line.replace(',', '')
		line = line.replace("[","")
		line = line.replace("]","")
		line = line.replace("'","")
		best_features.append(line.split())
	return best_features


def list_features_local(list_all_features):
	list_features_remove = deepcopy(list_all_features)

	for i in range(len(list_all_features)):
		if(('Norm' in list_all_features[i]) or ('com' in list_all_features[i])):
			del list_features_remove[list_features_remove.index(list_all_features[i])]

	return list_features_remove


def load_data_from_dump(path_data):
	with open(path_data + 'save_data_dump.pkl', 'rb') as input:
		data_win = pickle.load(input)
	with open(path_data + 'save_labels_dump.pkl', 'rb') as input:
		real_labels = pickle.load(input)
	with open(path_data + 'save_liststates_dump.pkl', 'rb') as input:
		list_states = pickle.load(input)
	with open(path_data + 'save_listfeatures_dump.pkl', 'rb') as input:
		list_features = pickle.load(input)
	if path.isfile(path_data + 'save_listsequence_dump.pkl'):	
		with open(path_data + 'save_listsequence_dump.pkl', 'rb') as input:
			path_data = pickle.load(input)
		return list_sequence, data_win, real_labels, list_states, list_features
	
	else:
		return data_win, real_labels, list_states, list_features

def save_data_to_dump(path, list_seq, data, labels, list_states, list_features):
	if not os.path.exists(path):
		os.mkdir(path)

	pickle.dump(data, open(path +"save_data_dump.pkl", "wb" ) )
	pickle.dump(labels, open(path + "save_labels_dump.pkl", "wb" ) )
	pickle.dump(list_states, open(path + "save_liststates_dump.pkl", "wb" ) )
	pickle.dump(list_features, open(path + "save_listfeatures_dump.pkl", "wb" ) )
	pickle.dump(list_seq, open(path + "save_listsequence_dump.pkl", "wb" ) )


def reduce_data_to_features(data_all, list_features_total, list_features_final):
	reduce_data = []
	for data in data_all:
		df = pd.DataFrame(data)
		df.columns = list_features_total
		reduce_data.append(df[list_features_final].values)
	return reduce_data







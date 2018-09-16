import numpy as np
from hmmlearn import hmm
import tools as tools
import os
from sklearn.externals import joblib
from lxml import etree as ET

def relative_dir(abs_path):
  return os.path.realpath(
      os.path.join(os.path.dirname(abs_path),
      '../../../online_recognition/src/'))

def addpack(abs_path):
  relative = relative_dir(abs_path)
  if relative not in sys.path:
    sys.path.append(relative)

class ModelHMM():
	""" Hiden Markov Model classifier class

	Attributes:
	n_states: int containing the number of states describing the model
	list_states: list of string containing the labels of each state
	list_feature: list (1 x n_features) of string containing the name of each feature
	dim_feature: list (1 x n_features) of int containing the dimension of each feature
	trans_mat: array (n_states x n_states) containing the transition probabilities
	start_prob: array (1 x n_states) containing the initial probabilities
	emission_means: array (n_states x n_features) containing the mean of the Gaussian 
		distribution representing the emission probabilities of the observations
	emission_covars: array (n_states x n_features x n_features) containing the covariance
		 matrix of the Gaussian distribution representing the emission probabilities of 
		 the observations
	"""
	def __init__(self):
		super(ModelHMM, self).__init__()
		self.n_states = 0
		self.list_states = []
		self.list_features = []
		self.dim_features = []
		self.trans_mat = []
		self.emission_means = []
		self.emission_covars = []
		self.start_prob = []


	def train(self, data, real_labels, list_features, dim_features):
		""" Train a supervised HMM classifier based on the data and labels in input

		input:
		data: a list (n_seq) of array (n_feature x length of sequence) containing the data used to train the model
		real_labels: a list (n_seq) of array (1 x length of sequence) containing the annotated labels of the state
		list_feature: a list containaing the name of the features used to train the model
		dim_feature: a list containing the dimension of each feature

		The parameters of the HMM trained are:
		startprob_: an array (1 x n_state) containing the initial state probabilities
		transmat_: an array (n_state x n_state) containing the transition matrix probability
		And the Gaussian distribution representing the emission probabilities represented by:
		means_: an array (n_state x n_feature) containing for each state the means of the multivariate Gaussian function
		covars_: an array (n_state x n_feature x n_feature) containing for each state the covariance matrix 
			of the multivariate Gaussian function
		"""
		self.n_seq = len(data)
		self.list_features = list_features
		self.dim_features = dim_features
		self.n_feature = int(sum(dim_features))


		# Concatenate all the sequence in one and create a vector with the length of each sequence
		obs = []
		obs = data[0]
		lengths = []
		lengths.append(len(data[0]))
		labels = real_labels[0]

		for i in range(1, self.n_seq):
			obs = np.concatenate([obs, data[i]])
			lengths.append(len(data[i]))
			labels = np.concatenate([labels, real_labels[i]])

		# Get the list and number of states
		self.list_states, labels = np.unique(labels, return_inverse=True)
		self.n_states = len(self.list_states)

		self.model = hmm.GaussianHMM(n_components=self.n_states, covariance_type="full")

		Y = labels.reshape(-1, 1) == np.arange(len(self.list_states))
		end = np.cumsum(lengths)
		start = end - lengths



		# Compute the initial probabilities
		init_prob = Y[start].sum(axis=0)/Y[start].sum()
		# init_prob = np.ones(self.n_states)/self.n_states

		# Compute the transition matrix probabilities
		trans_prob = np.zeros((self.n_states, self.n_states)).astype(int)
		for i in range(1, len(labels)):
			trans_prob[labels[i-1], labels[i]] += 1


		
		trans_prob = trans_prob/np.sum(trans_prob, axis=0)

		# Compute the emission distribution
		Mu, covars = tools.mean_and_cov(obs, labels, self.n_states, self.list_features)

		# Update the parameters of the model
		self.model.startprob_ = init_prob
		self.model.transmat_ = trans_prob.T
		self.model.means_ = Mu
		self.model.covars_ = covars

		return

	def predict_states(self, obs):
		""" Return for a sequence of observation a sequence of predicted labels

		"""
		return self.model.predict(obs)

	def score_samples(self, obs):
		""" 
		Return for a sequence of observation the probability distribution on each state
		"""
		return self.model.score_samples(obs)[1]


	def save_model(self, path, name_model, name_sequences):
		if(not(os.path.isdir(path))):
			os.makedirs(path)

		joblib.dump(self.model, path +'/' + name_model)

		root = ET.Element('HMM-Model')
		tree = ET.ElementTree(root)

		model = ET.SubElement(root, 'model', name_model = name_model, name_sequences = name_sequences)


		list_features = ET.SubElement(model, 'features')

		sub_feature_list = []

		# for f,d, index in zip(self.list_features, self.dim_features, range(len(self.list_features))):
		# 	name_feature = f.split('_')
		# 	if(len(name_feature) == 1):
		# 		feature = ET.SubElement(list_features, 'feature', index=str(index), label=f, dimension=str(d))
		# 	else:
		# 		category = name_feature[0]
		# 		if(category in sub_feature_list):
		# 			sub_features = ET.SubElement(list_features, 'sub_feature', label=category)
		# 			sub_feature_list.append(category)
		# 		else:
		# 			sub_features = ET.find(category)

				# feature = ET.SubElement(sub_features, 'feature', index=str(index), label=name_feature[1], dimension=str(d))
			
		states = ET.SubElement(model, 'states')
		for s, index in zip(self.list_states, range(self.n_states)):
			state = ET.SubElement(states, 'state', index=str(index), label=s)

		features = ET.SubElement(model, 'features')




		name_file = path + '/' +  name_model + '.xml'
		tree.write(name_file, pretty_print=True, xml_declaration=True,   encoding="utf-8")
		return


	def load_model(self, model_file):
		""" 
		Load the parameters from the configuration file to set the parameters of the model
		"""
		self.model = joblib.load(model_file)

		tree = ET.parse(model_file + '.xml')
		print(tree)
		data = next(tree.iterfind('model'))
		states = list(next(data.iterfind('states')))
		self.list_states = []

		for state in states:
			self.list_states.append(state.get('label'))
		return

	def test_model(self, data_test):
		"""
		This function return a list of labels infered from the sequences of observation in input
		"""
		predict_labels = [[]]
		proba = []

		for data, i in zip(data_test, range(len(data_test))):
			labels = self.predict_states(data)
			proba.append(self.score_samples(data))

			for j in range(len(labels)):
				predict_labels[i].append(self.list_states[labels[j]])

			if(len(predict_labels) < len(data_test)):
				predict_labels.append([])

		return predict_labels, proba



	def get_list_states(self):
		return self.list_states

	def get_n_states(self):
		return self.n_states

	def get_list_features(self):
		return self.list_features

	def get_dim_features(self):
		return self.dim_features

	def get_trans_mat(self):
		return self.model.transmat_

	def get_start_prob(self):
		return start_prob

	def get_emission_prob(self):
		return self.model.means_, self.model.covars_

	def get_model(self):
		return self.model



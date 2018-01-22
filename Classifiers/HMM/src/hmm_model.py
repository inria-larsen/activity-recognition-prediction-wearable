import numpy as np
from hmmlearn import hmm
import src.tools as tools

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


	def train(self, data, real_labels, list_features, dim_features, index_sequences):
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
		self.n_feature = sum(dim_features)
		self.index_sequences = index_sequences



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

		# Compute the transition matrix probabilities
		trans_prob = np.zeros((self.n_states, self.n_states)).astype(int)
		for i in range(1, len(labels)):
			trans_prob[labels[i-1], labels[i]] += 1
		
		trans_prob = trans_prob/np.sum(trans_prob, axis=0)

		# Compute the emission distribution
		Mu, covars = tools.mean_and_cov(obs, labels, self.n_states, self.n_feature)

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
		""" Return for a sequence of observation the probability distribution on each state

		"""
		return self.model.score_samples(obs)[1]


	def save_model(self):
		return


	def load_model(self, config_file):
		""" Load the parameters from the configuration file to set the parameters of the model


		"""
		return



	def get_list_states(self):
		return self.list_states

	def get_n_states(self):
		return self.n_states

	def get_list_features(self):
		return self.list_features

	def get_dim_features(self):
		return self.dim_features

	def get_trans_mat(self):
		return trans_mat

	def get_start_prob(self):
		return start_prob

	def get_emission_prob(self):
		return emission_means, emission_covars

	def get_model(self):
		return self.model



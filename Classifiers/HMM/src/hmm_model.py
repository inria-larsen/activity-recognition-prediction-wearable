import numpy as np
from hmmlearn import hmm

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



	def load(self, config_file):
		""" Load the parameters from the configuration file to set the parameters of the model


		"""
		return

	def train(self, data, lenghts):

		return


	def predict_states(self, data):
		return


	def compute_scores(self, data):
		return


	def save(self):
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



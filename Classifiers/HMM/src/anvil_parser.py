from lxml import etree
import numpy as np


class anvil_tree():
	"""
	This is a parser to get information from the Anvil files.
	"""

	def __init__(self, path):
		"""
		Constructor of the anvil_tree.
		Take in input the path where the file is located
		"""
		tree = etree.parse(path)
		self.data = list(next(tree.iterfind('body')))


	def get_data(self, name_track = '0'):
		"""
		Ouput: Return the label, start time and end time of all actions in the sequence.
		Input: name_track is a string corresponding to the track you want to get data from.
		"""

		if(name_track == '0'):
			track = self.data[0]
		else:
			for tr in self.data:
				if(tr.get('name') == name_track):
					track = tr

		self.start = []
		self.end = []
		self.label = []

		for el in track.iter('el'):
			self.start.append(float(el.get('start')))
			self.end.append(float(el.get('end')))
			self.label.append(next(el.iterfind('attribute')).text)

		return self.label, self.start, self.end

	def get_list_states(self):
		self.list_states = []
		for action in self.label:
			if(action not in self.list_states):
				self.list_states.append(action)
		return self.list_states






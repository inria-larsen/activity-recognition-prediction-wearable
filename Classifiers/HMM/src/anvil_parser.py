from lxml import etree
import numpy as np


class anvil_tree():

	def __init__(self, path):
		"""
		Constructor of the anvil_tree.
		Take in input the path where the file is located
		"""
		tree = etree.parse(path)
		self.data = next(tree.iterfind('body'))


	def get_data(self):
		"""
		Return the label, start time and end time of all actions in the sequence
		"""
		list_act = list(next(self.data.iterfind('track')))
		self.start = []
		self.end = []
		self.label = []
		for el in list_act:
			self.start.append(float(el.get('start'))*1000)
			self.end.append(float(el.get('end'))*1000)
			self.label.append(next(el.iterfind('attribute')).text)
		return self.label, self.start, self.end

	def get_list_states(self):
		self.list_states = []
		for action in self.label:
			if(action not in self.list_states):
				self.list_states.append(action)
		return self.list_states





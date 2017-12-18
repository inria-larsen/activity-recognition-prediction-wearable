from lxml import etree
import numpy as np
import matplotlib.pyplot as plt
import sys

class mvnx_tree():
	"""
	Utility to parse mnvx (xml) file from xsens and access data
	"""

	prefix = '{http://www.xsens.com/mvn/mvnx}' # the prefix used in mvnx file

	def __init__(self, path):
		"""
		Constructor of the mvnx_tree.
		Take in input the path where the file is located
		"""
		try:
			mvnx_tree = etree.parse(path)
			self.data = next(mvnx_tree.iterfind(self.prefix+'subject'))
		except OSError as e:
			raise(e)

	def get_data(self, tag):
		try:
			frame_list = list(next(self.data.iterfind(self.prefix+'frames')))
			data = []
			for frame in frame_list:
				if 	frame.get('type')=='normal':
					data.append(next(frame.iterfind(self.prefix+tag)).text.split())
			data = np.asarray(data)
			return(data.astype(np.float))
		except:
			raise NameError('Tag ', tag, ' does not exist')

	def get_timestamp(self):
		frame_list = list(next(self.data.iterfind(self.prefix+'frames')))
		timestamp = []
		for frame in frame_list:
			if 	frame.get('type')=='normal':
				timestamp.append(int(frame.get('time')))
		timestamp = np.asarray(timestamp)
		return(timestamp.T)

	def get_timestamp_ms(self):
		frame_list = list(next(self.data.iterfind(self.prefix+'frames')))
		timestamp = []
		for frame in frame_list:
			if 	frame.get('type')=='normal':
				timestamp.append(int(frame.get('ms')))
		timestamp = np.asarray(timestamp)
		return(timestamp.T/1000)

	def get_list(self, tag):
		return(list(next(self.data.iterfind(self.prefix+tag))))

	def get_id_segment(self, segment_name):
		segments = self.get_list('segments')
		id = 0
		while segments[id].get('label')!=segment_name:
			id+=1
		return id

	def get_name_segment(self, id):
		segments = self.get_list('segments')
		return segments[id].get('label')

	def get_id_joint(self, joint_name):
		joints = self.get_list('joints')
		id = 0
		while joints[id].get('label')!=joint_name:
			id+=1
		return id

	def get_name_joint(self, id):
		try:
			joints = self.get_list('joints')
			if(id>=len(joints) or id<0):
				raise ValueError("Id out of bound")
			return joints[id].get('label')
		except ValueError as ex:
			return


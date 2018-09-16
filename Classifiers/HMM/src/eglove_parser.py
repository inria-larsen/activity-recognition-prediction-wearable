from lxml import etree
import numpy as np

class eglove_tree():

	def __init__(self, path):
		"""
		Constructor of the eglove parser.
		Take in input the path where the file is located
		"""
		tree = etree.parse(path)
		self.header = next(tree.iterfind('header'))
		self.signals = list(next(tree.iterfind('signals')))
		self.frames = list(next(tree.iterfind('frames')))

	def get_data_by_signal(self, list_signal):
		"""
		Input: list of signal labels
		Ouput: an np.array with the related data
		"""
		data = np.zeros((len(self.frames), len(list_signal)))
		i = 0
		for signal in list_signal:
			id_signal = self.get_id_signal(signal)
			for frame in self.frames:
				id_frame = frame.get('index')
				data[int(id_frame), i] = next(frame.iterfind(signal)).text
			i += 1
		return data


	def get_all_data(self):
		"""
		Input: list of signal labels
		Ouput: an np.array with the related data
		"""
		data = np.zeros((len(self.frames), len(self.signals)))
		i = 0
		for signal in self.signals:
			label_signal = signal.get('label')
			print(label_signal)
			id_signal = self.get_id_signal(label_signal)
			for frame in self.frames:
				id_frame = frame.get('index')
				data[int(id_frame), i] = next(frame.iterfind(label_signal)).text
			i += 1
		return data


	def get_timestamp(self):
		timestamp = []
		for frame in self.frames:
			timestamp.append(float(frame.get('ts')))
		return timestamp

	def get_time(self):
		time = []
		for frame in self.frames:
			time.append(frame.get('time'))
		return time

	def get_id_signal(self, signal_label):
		for signal in self.signals:
			if(signal_label == signal.get('label')):
				return signal.get('index')
		return
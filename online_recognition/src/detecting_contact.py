import yarp
import sys
import numpy as np
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Classifiers/HMM/src/')))

from hmm_model import ModelHMM

import warnings
warnings.filterwarnings("ignore")


class DetectionContact(object):
	def __init__(self, rf):
		self.input_port = '/processing/eglove/data/Forces:o'

		self.port = yarp.BufferedPortBottle()
		self.port.open('/DetectionContact')

		path_model = rf.find('path_model').toString()
		name_model = rf.find('name_model').toString()

		self.model = ModelHMM()
		self.model.load_model(path_model + '/' + name_model)
		self.list_states = self.model.get_list_states()
		self.buffer_model = []
		self.obs = []
		print(self.list_states)

		yarp.Network.connect(self.input_port, self.port.getName())

	def update(self):
		b_in = self.port.read()
		data = b_in.toString().split(' ')

		# data = self.cback[i].get_data()

		if(len(data)>0):
			self.buffer_model = data
			# self.flag_model[i] = 1

			


			del data[0]
			data_model = list(map(float, data))

			self.obs.append(data_model)

			if(len(self.obs) > 500):
				del self.obs[0]


		if(len(self.obs) > 10):
			state = self.model.predict_states(self.obs)
			b_state = self.port.prepare()
			b_state.clear()
			b_state.addInt(1)
			index_state = int(state[-1])
			b_state.addInt(index_state)
			b_state.addString(self.list_states[index_state])
			self.port.write()


def main():
	yarp.Network.init()
	rf = yarp.ResourceFinder()
	rf.setDefaultContext("online_recognition")
	rf.setDefaultConfigFile("eglove_only.ini")
	rf.configure(sys.argv)

	contact_module = DetectionContact(rf)

	while(True):
		try:
			contact_module.update()
		except KeyboardInterrupt:
			break



if __name__ == '__main__':
    main()
		
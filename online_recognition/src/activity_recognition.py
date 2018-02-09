#!/usr/bin/python3

import yarp
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Classifiers/HMM/src/')))

from hmm_model import ModelHMM

import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


class ActivityRecognitionModule(yarp.RFModule):
	def __init__(self):
		yarp.RFModule.__init__(self)
		self.handlerPort = yarp.Port()

	def configure(self, rf):
		self.handlerPort.open("/ActivityRecognitionModule")
		self.attach(self.handlerPort)

		self.list_port = []
		self.cback = []

		path_model = rf.find('path_model').toString()
		name_model = rf.find('name_model').toString()
		self.statePort = yarp.BufferedPortBottle()
		self.statePort.open('/activity_recognition/state:o')
		self.probPort = yarp.BufferedPortBottle()
		self.probPort.open('/activity_recognition/probabilities:o')
		self.model = ModelHMM()
		self.model.load_model(path_model + '/' + name_model)
		self.list_states = self.model.get_list_states()

		print(self.list_states)

		size_buffer = int(rf.find('size_buffer').toString())

		signals = rf.findGroup("Signals").tail().toString().replace(')', '').replace('(', '').split(' ')
		nb_port = int(len(signals))
		nb_active_port = 0

		for signal in signals:
			info_signal = rf.findGroup(signal)
			is_enabled = int(info_signal.find('enable').toString())

			if(is_enabled):
				list_items = info_signal.findGroup('list').tail().toString().split(' ')
				input_port_name = info_signal.find('output_port').toString()

				if(input_port_name == ''):
					input_port_name = info_signal.find('input_port').toString()


				if((list_items[0] == 'all') or (list_items[0] == '')):
					self.list_port.append(yarp.BufferedPortBottle())
					self.list_port[nb_active_port].open("/activity_recognition" + input_port_name + ':i')
					self.cback.append(CallbackData(size_buffer))
					self.list_port[nb_active_port].useCallback(self.cback[nb_active_port])
					yarp.Network.connect("/processing" + input_port_name + ':o', self.list_port[nb_active_port].getName())
					nb_active_port += 1

				else:
					for item in list_items:
						self.list_port.append(yarp.BufferedPortBottle())
						self.list_port[nb_active_port].open("/activity_recognition" + input_port_name + '/' + item + ':i')
						self.cback.append(CallbackData(size_buffer))
						self.list_port[nb_active_port].useCallback(self.cback[nb_active_port])
						yarp.Network.connect("/processing" + input_port_name + '/' + item  + ':o', self.list_port[nb_active_port].getName())
						nb_active_port += 1



		self.obs = []

		return True

	def close(self):
		for port in self.list_port:
			port.close()
		return True

	def updateModule(self):
		data_model = []
		index = 0
		received_data = 0
		for port, i in zip(self.list_port, range(len(self.list_port))):
			data = self.cback[i].get_data()
			if(len(data)>0):
				received_data += 1
				dimension = int(data[0])
				del data[0]

				if(i == 0):
					data_model = data
				else:
					data_model = np.concatenate((data_model, data))

		if(received_data == len(self.list_port)):
			self.obs.append(data_model)

			if(len(self.obs) > 10):
				state = self.model.predict_states(self.obs)
				b_state = self.statePort.prepare()
				b_state.clear()
				b_state.addInt(1)
				index_state = int(state[-1])
				b_state.addInt(index_state)
				b_state.addString(self.list_states[index_state])
				self.statePort.write()

				probabilities = self.model.score_samples(self.obs)
				b_prob = self.probPort.prepare()
				b_prob.clear()

				for prob, index in zip(probabilities[-1], range(len(probabilities[-1]))):
					b_prob.addString(self.list_states[index])
					b_prob.addDouble(prob)
				self.probPort.write()

		return True

	def getPeriod(self):
		return 0.01


class CallbackData(yarp.BottleCallback):
	def __init__(self, size_buffer):
		yarp.BottleCallback.__init__(self)
		self.port = yarp.Port()
		self.buffer = []
		self.size_buffer = size_buffer

	def onRead(self, bot, *args, **kwargs):
		data = bot.toString().split(' ')
		value = list(map(float, data))
		self.buffer = value
		if(len(self.buffer) > self.size_buffer):
			del self.buffer[0]
		return value

	def get_data(self):
		data = self.buffer
		self.buffer = []
		return data


if __name__=="__main__":
	yarp.Network.init()

	rf = yarp.ResourceFinder()
	rf.setVerbose(True)
	rf.setDefaultContext("online_recognition")
	rf.setDefaultConfigFile("default.ini")
	rf.configure(sys.argv)

	mod_recognition = ActivityRecognitionModule()
	mod_recognition.configure(rf)

	while(True):
		try:
			yarp.Time.delay(mod_recognition.getPeriod())
			mod_recognition.updateModule();
		except KeyboardInterrupt:
			break

	mod_recognition.close()
	yarp.Network.fini();

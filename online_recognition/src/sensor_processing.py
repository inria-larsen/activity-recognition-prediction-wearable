#!/usr/bin/python3

import yarp
import sys
import matplotlib.pyplot as plt
import numpy as np


class SensorProcessingModule(yarp.RFModule):
	def __init__(self):
		yarp.RFModule.__init__(self)
		self.handlerPort = yarp.Port()

	def configure(self, rf):
		self.handlerPort.open("/SensorProcessingModule")
		self.attach(self.handlerPort)
		self.list_port = []
		self.cback = []

		size_buffer = int(rf.find('size_buffer').toString())

		self.window_size = float(rf.find('slidding_window_size').toString())


		signals = rf.findGroup("Signals").tail().toString().replace(')', '').replace('(', '').split(' ')
		print(signals)
		port_information = signals
		print(port_information)

		nb_port = int(len(signals))
		nb_active_port = 0

		self.buffer = []

		for signal in signals:
			info_signal = rf.findGroup(signal)
			is_enabled = int(info_signal.find('enable').toString())

			if(is_enabled):
				name_port = info_signal.find('name_port').toString()
				self.list_port.append(yarp.BufferedPortBottle())
				self.list_port[nb_active_port].open("/processing" + name_port)
				self.cback.append(CallbackData(size_buffer))
				self.list_port[nb_active_port].useCallback(self.cback[nb_active_port]) 
				nb_active_port += 1
				self.buffer.append([])


		self.clock = yarp.Time.now()


		return True

	def close(self):
		for port in self.list_port:
			port.close()
		return True

	def updateModule(self):
		for port, i in zip(self.list_port, range(len(self.list_port))):
				data = self.cback[i].get_data()
				if(len(data)>0):
					for j in range(len(data)):
						self.buffer[i].append(data[i])

		current_time = yarp.Time.now()

		# Slidding Window
		if((current_time - self.clock) >= self.window_size/2):
			self.clock = current_time
			for port, i in zip(self.list_port, range(len(self.list_port))):
				if(len(self.buffer[i]) > 0):
					length = int(len(self.buffer[i])/2)
					output = np.mean(np.asarray(self.buffer[i]), axis = 0)
					dimension = np.shape(output)[0]
					b_out = port.prepare()
					b_out.clear()
					b_out.addInt(dimension)
					for dim in range(dimension):
						b_out.addDouble(output[dim])
					port.write()


					del self.buffer[i][0:length]

		return True

	def getPeriod(self):
		return 0.001


class CallbackData(yarp.BottleCallback):
	def __init__(self, size_buffer):
		yarp.BottleCallback.__init__(self)
		self.port = yarp.Port()
		self.buffer = []
		self.size_buffer = size_buffer

	def onRead(self, bot, *args, **kwargs):
		data = bot.toString().split(' ')
		value = list(map(float, data))
		self.buffer.append(value)
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

	mod_sensor = SensorProcessingModule()
	mod_sensor.configure(rf)

	while(True):
		try:
			yarp.Time.delay(mod_sensor.getPeriod())
			mod_sensor.updateModule();
		except KeyboardInterrupt:
			break

	mod_sensor.close()
	yarp.Network.fini();

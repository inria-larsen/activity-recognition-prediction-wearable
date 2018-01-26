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

		signals = rf.findGroup("Signals").tail().toString().replace('"', '').replace(')', '').replace('(', '')
		port_information = signals.split(' ')
		
		nb_port = int(len(port_information)/3)
		nb_active_port = 0

		for i in range(nb_port):
			id_port = i*(nb_port-1)
			if(int(port_information[id_port + 1])):
				print(port_information[id_port+2], nb_active_port)
				self.list_port.append(yarp.BufferedPortBottle())
				self.list_port[nb_active_port].open("/processing" + port_information[id_port + 2])
				self.cback.append(CallbackData(size_buffer))
				self.list_port[nb_active_port].useCallback(self.cback[nb_active_port]) 
				nb_active_port += 1

		return True

	def close(self):
		for port in self.list_port:
			port.close()
		return True

	def updateModule(self):
		for port, i in zip(self.list_port, range(len(self.list_port))):
			data = self.cback[i].get_data()
			if(len(data)>0):
				T, dim = np.shape(data)
				b_out = port.prepare()
				b_out.clear()
				b_out.addInt(T)
				b_out.addInt(dim)
				for m in range(T):
					for n in range(dim):
						b_out.addDouble(data[m][n])
				
				port.write()

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
	rf.setDefaultContext("myContext")
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

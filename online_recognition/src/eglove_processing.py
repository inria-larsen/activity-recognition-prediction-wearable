#!/usr/bin/python3

import yarp
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

class SensorProcessingModule(yarp.RFModule):
	def __init__(self):
		yarp.RFModule.__init__(self)
		self.handlerPort = yarp.Port()

	def configure(self, rf):
		size_buffer = int(rf.find('size_buffer').toString())
		self.handlerPort.open("/EgloveProcessingModule")
		self.attach(self.handlerPort)
		self.input_port = yarp.BufferedPortBottle()
		self.output_port = []
		self.cback = CallbackData(size_buffer)
		self.list_name_ports = []
		self.related_port = []
		self.norm_list = []

		self.window_size = float(rf.find('slidding_window_size').toString())

		signals = rf.findGroup("Signals").tail().toString().replace(')', '').replace('(', '').split(' ')
		nb_port = int(len(signals))
		nb_output_port = 0

		self.buffer = []

		signal = "eglove"
		info_signal = rf.findGroup(signal)
		is_enabled = int(info_signal.find('enable').toString())

		if(is_enabled):
			list_items = info_signal.findGroup('list').tail().toString().split(' ')
			input_port_name = info_signal.find('input_port').toString()
			output_port_name = info_signal.find('output_port').toString()
			if(output_port_name == ''):
				output_port_name = input_port_name


			if((list_items[0] == 'all') or (list_items[0] == '')):
				if(not(input_port_name in self.list_name_ports)):
					self.input_port.open("/processing" + input_port_name +':i')			
					self.list_name_ports.append(input_port_name)
					self.input_port.useCallback(self.cback)
				
				self.output_port.append(yarp.BufferedPortBottle())
				self.output_port[nb_output_port].open("/processing" + output_port_name + ':o')
				self.related_port.append([input_port_name, 'all'])
				nb_output_port += 1

				is_norm = info_signal.find('norm').toString()
				if(is_norm == '' or is_norm == '0'):
					self.norm_list.append(0)
				else:
					self.norm_list.append(int(is_norm))

			else:
				for item in list_items:
					if(not(input_port_name in self.list_name_ports)):
						self.input_port.open("/processing" + input_port_name + ':i')
						self.input_port.useCallback(self.cback)
						self.list_name_ports.append(input_port_name)

					self.output_port.append(yarp.BufferedPortBottle())
					self.output_port[nb_output_port].open("/processing" + output_port_name + '/' + item + ':o')
					related_items = info_signal.find("related_items").toString()

					if(related_items == ''):
						related_data = info_signal.find('related_data').toString()
						info_related_data = rf.findGroup(related_data)
						related_items = info_related_data.find("related_items").toString()

					is_norm = info_signal.find('norm').toString()
					if(is_norm == '' or is_norm == '0'):
						self.norm_list.append(0)
					else:
						self.norm_list.append(int(is_norm))

					id_item = int(rf.findGroup(related_items).find(item).toString())
					nb_items = int(rf.findGroup(related_items).find('Total').toString())
					nb_output_port += 1
					self.related_port.append([input_port_name, id_item, nb_items])

		self.clock = yarp.Time.now()

		return True

	def close(self):
		self.input_port.close()
		return True

	def updateModule(self):
		received_data = self.read_input_ports()

		current_time = yarp.Time.now()
		
		# Slidding Window
		if((current_time - self.clock) >= self.window_size/2):
			self.clock = current_time

			# Get the data from each input port corresponding to the window
		if(received_data == 0 and len(self.buffer) > 0):
			del self.buffer[0]

		if(len(self.buffer) > 0):
			length = int(len(self.buffer)/2)

			N = len(self.buffer)
			
			data_output = self.buffer

			# Check all ouput port to send data
			for out_port, id_ouput in zip(self.output_port, range(len(self.output_port))):
				# Check which data send to the output port (segments/joints)
				id_items = self.related_port[id_ouput][1]
				nb_items = self.related_port[id_ouput][2]

				if(id_items == 'all'):
					id_items = [0, len(data_output[0])]

# 				
				out = np.asarray(data_output)

				# Compute the average signals on the sliding window
				output = np.mean(np.asarray(out[:,id_items:id_items+nb_items]), axis = 0)

				if(self.norm_list[id_ouput]):
					output = np.expand_dims(output, axis=1)
					output = np.linalg.norm(output, axis = 0)

				b_out = out_port.prepare()
				b_out.clear()
				b_out.addInt(len(output))
				for i in range(len(output)):
					b_out.addDouble(output[i])
				out_port.write()
			
			del self.buffer[0:length]

		return True


	def read_input_ports(self):
		received_data = 0
		# for port, i in zip(self.input_port, range(len(self.input_port))):

		data = self.cback.get_data()
		
		if(len(data)>0):
			received_data = 1
			for j in range(len(data)):
				self.buffer.append(data[j])

		return received_data

	def getPeriod(self):
		return 0.001


class CallbackData(yarp.BottleCallback):
	def __init__(self, size_buffer):
		yarp.BottleCallback.__init__(self)
		self.port = yarp.Port()
		self.buffer = []
		self.size_buffer = size_buffer

	def onRead(self, bot, *args, **kwargs):
		data = bot.toString().replace(')', '').replace('(', '').split()
		

		value = list(map(float, data))
		for i in range(0, len(value), 8):
			self.buffer.append(value[i:i+7])
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
	rf.setDefaultConfigFile("eglove_only.ini")
	rf.configure(sys.argv)

	mod_sensor = SensorProcessingModule()
	mod_sensor.configure(rf)
	# mod_sensor.runModule(rf)

	while(True):
		try:
			yarp.Time.delay(mod_sensor.getPeriod())
			mod_sensor.updateModule()
			
		except KeyboardInterrupt:
			break

	mod_sensor.close()
	yarp.Network.fini();

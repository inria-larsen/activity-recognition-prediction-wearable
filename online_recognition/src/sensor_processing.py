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
		self.input_port = []
		self.output_port = []
		self.cback = []
		self.diff_order_list = []
		self.norm_list = []
		self.list_name_ports = []
		self.related_port = []

		size_buffer = int(rf.find('size_buffer').toString())

		self.window_size = float(rf.find('slidding_window_size').toString())

		signals = rf.findGroup("Signals").tail().toString().replace(')', '').replace('(', '').split(' ')
		nb_port = int(len(signals))
		nb_active_port = 0
		nb_output_port = 0

		self.buffer = []

		for signal in signals:
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
						self.input_port.append(yarp.BufferedPortBottle())
						self.input_port[nb_active_port].open("/processing" + input_port_name +':i')			
						self.list_name_ports.append(input_port_name)
						self.cback.append(CallbackData(size_buffer))
						self.input_port[nb_active_port].useCallback(self.cback[nb_active_port])
					
					nb_active_port += 1
					self.output_port.append(yarp.BufferedPortBottle())
					self.output_port[nb_output_port].open("/processing" + output_port_name + ':o')
					self.related_port.append([input_port_name, 'all'])
					nb_output_port += 1
					self.buffer.append([])
					order_diff = info_signal.find('diff_order').toString()
					if(order_diff == '' or order_diff == '0'):
						self.diff_order_list.append(0)
					else:
						self.diff_order_list.append(int(order_diff))

				else:
					for item in list_items:
						if(not(input_port_name in self.list_name_ports)):
							self.input_port.append(yarp.BufferedPortBottle())
							self.input_port[nb_active_port].open("/processing" + input_port_name + ':i')
							self.cback.append(CallbackData(size_buffer))
							self.input_port[nb_active_port].useCallback(self.cback[nb_active_port])
							self.list_name_ports.append(input_port_name)
							self.buffer.append([])
							nb_active_port += 1

						order_diff = info_signal.find('diff_order').toString()
						if(order_diff == '' or order_diff == '0'):
							self.diff_order_list.append(0)
						else:
							self.diff_order_list.append(int(order_diff))

						is_norm = info_signal.find('norm').toString()
						if(is_norm == '' or is_norm == '0'):
							self.norm_list.append(0)
						else:
							self.norm_list.append(int(is_norm))

						self.output_port.append(yarp.BufferedPortBottle())
						self.output_port[nb_output_port].open("/processing" + output_port_name + '/' + item + ':o')
						related_items = info_signal.find("related_items").toString()
						id_item = int(rf.findGroup(related_items).find(item).toString())
						nb_output_port += 1
						self.related_port.append([input_port_name, np.arange(id_item, id_item + 3)])

		self.clock = yarp.Time.now()

		return True

	def close(self):
		for port in self.input_port:
			port.close()
		return True

	def updateModule(self):
		received_data = self.read_input_ports()

		current_time = yarp.Time.now()
		
		# Slidding Window
		if((current_time - self.clock) >= self.window_size/2):
			self.clock = current_time

			# Get the data from each input port corresponding to the window
			for in_port, id_input in zip(self.input_port, range(len(self.input_port))):			
				if(received_data[id_input] == 0 and len(self.buffer[id_input]) > 0):
					del self.buffer[id_input][0]

				if(len(self.buffer[id_input]) > 0):
					length = int(len(self.buffer[id_input])/2)
					
					data_output = self.buffer[id_input]

					# Check all ouput port to send data
					for out_port, id_ouput in zip(self.output_port, range(len(self.output_port))):
						if(self.list_name_ports.index(self.related_port[id_ouput][0]) == id_input):

							# Check which data send to the output port (segments/joints)
							id_items = self.related_port[id_ouput][1]
							if(id_items == 'all'):
								id_items = [0, len(data_output[0])]

							# Derive the signal to extract velocity or acceleration
							if(self.diff_order_list[id_ouput] > 0):
								out = np.diff(np.asarray(data_output), self.diff_order_list[id_ouput], axis=0)
								for j in range(self.diff_order_list[id_ouput]):
									out = np.insert(out, 0, 0, axis=0)
							else:
								out = np.asarray(data_output)

							# Compute the average signals on the sliding window
							output = np.mean(np.asarray(out[:,id_items[0]:id_items[-1]+1]), axis = 0)

							# Send data to the ouput port
							dimension = np.shape(output)[0]
							b_out = out_port.prepare()
							b_out.clear()
							b_out.addInt(dimension)
							for dim in range(dimension):
								b_out.addDouble(output[dim])
							out_port.write()

							print(out_port.getName(), np.shape(output))
						
					del self.buffer[id_input][0:length]

		return True


	def read_input_ports(self):
		received_data = np.zeros(len(self.input_port))
		for port, i in zip(self.input_port, range(len(self.input_port))):

			data = self.cback[i].get_data()

			if(len(data)>0):
				received_data[i] = 1
				for j in range(len(data)):
					self.buffer[i].append(data[j])
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
	# mod_sensor.runModule(rf)

	while(True):
		try:
			yarp.Time.delay(mod_sensor.getPeriod())
			mod_sensor.updateModule()
			
		except KeyboardInterrupt:
			break

	mod_sensor.close()
	yarp.Network.fini();

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
		self.handlerPort.open("/SensorProcessingModule")
		self.attach(self.handlerPort)
		self.input_port = []
		self.output_port = []
		self.cback = []
		self.diff_order_list = []
		self.norm_list = []
		self.normalize = []
		self.dist_com_list = []
		self.list_name_ports = []
		self.median_normalize = [[]]
		self.related_port = []
		self.count = 0
		self.current_com = []
		self.flag_timer = 0
		self.init_com = []

		self.port_init_com = yarp.BufferedPortBottle()
		self.port_init_com.open('/processing/init_com:i')
		self.cback_init = CallBackInitCom()
		self.port_init_com.useCallback(self.cback_init)

		size_buffer = int(rf.find('size_buffer').toString())

		self.window_size = float(rf.find('slidding_window_size').toString())

		signals = rf.findGroup("Signals").tail().toString().replace(')', '').replace('(', '').split(' ')
		nb_port = int(len(signals))
		nb_active_port = 0
		nb_output_port = 0

		self.buffer = []

		for signal in signals:
			info_signal = rf.findGroup(signal)
			print(signal)
			is_enabled = int(info_signal.find('enable').toString())

			if(signal == ('eglove')):
				continue

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

					is_norm = info_signal.find('norm').toString()
					if(is_norm == '' or is_norm == '0'):
						self.norm_list.append(0)
					else:
						self.norm_list.append(int(is_norm))

					normalize = info_signal.find('normalize').toString()
					if(normalize == '' or normalize == '0'):
						self.normalize.append([0])
					else:
						self.normalize.append([int(normalize), 0])


					dist_com = info_signal.find('dist_com').toString()
					if(dist_com == '' or dist_com == '0'):
						self.dist_com_list.append(0)
					else:
						self.dist_com_list.append(int(dist_com), 0)

				else:
					for item_name in list_items:
						item_carac = item_name.split('_')
						item = item_carac[0]

						print(item_carac)

						if len(item_carac) == 2:
							if item_carac[1] in ['x', 'q1']:
								dim_item = 0
							elif item_carac[1] in ['y', 'q2']:
								dim_item = 1
							elif item_carac[1] in ['z', 'q3']:
								dim_item = 2
							elif item_carac[1] in ['q4']:
								dim_item = 3
						else:
							dim_item = 0

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

						normalize = info_signal.find('normalize').toString()
						if(normalize == '' or normalize == '0'):
							self.normalize.append([0])
						else:
							self.normalize.append([int(normalize), 0])
						self.median_normalize.append([])

						dist_com = info_signal.find('dist_com').toString()
						if(dist_com == '' or dist_com == '0'):
							self.dist_com_list.append(0)
						else:
							self.dist_com_list.append(int(dist_com))


						self.output_port.append(yarp.BufferedPortBottle())
						self.output_port[nb_output_port].open("/processing" + output_port_name + '/' + item_name + ':o')
						related_items = info_signal.find("related_items").toString()

						id_item = int(rf.findGroup(related_items).find(item).toString())
						dimension = int(rf.findGroup(related_items).find('dimension').toString())

						id_item = id_item*dimension + dim_item
						dimension = 1

						nb_output_port += 1
						self.related_port.append([input_port_name, id_item, dimension])

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
		if((current_time - self.clock) >= self.window_size):
			initalization = self.cback_init.get_data()

			self.clock = current_time

			# Get the data from each input port corresponding to the window
			for in_port, id_input in zip(self.input_port, range(len(self.input_port))):	
				if(received_data[id_input] == 0 and len(self.buffer[id_input]) > 0):
					del self.buffer[id_input][0]

				if(len(self.buffer[id_input]) > 0):
					length = int(len(self.buffer[id_input])/2)
					
					data_output = self.buffer[id_input]

					if(in_port.getName() == '/processing/xsens/COM:i'):
						self.current_com = np.mean(np.asarray(data_output), axis = 0)
						if(self.count == 0 or initalization == 1):
							self.init_com = self.current_com

					# Check all ouput port to send data
					for out_port, id_ouput in zip(self.output_port, range(len(self.output_port))):

						if(self.list_name_ports.index(self.related_port[id_ouput][0]) == id_input):

							# Check which data send to the output port (segments/joints)
							id_items = self.related_port[id_ouput][1]

							if(id_items == 'all'):
								id_items = 0
								dimension = len(data_output[0])
							else:
								dimension = self.related_port[id_ouput][2]

							# Derive the signal to extract velocity or acceleration
							if(self.diff_order_list[id_ouput] > 0):
								out = np.diff(np.asarray(data_output), self.diff_order_list[id_ouput], axis=0)
								for j in range(self.diff_order_list[id_ouput]):
									out = np.insert(out, 0, 0, axis=0)

								order = 6
								fs = 240
								nyq = 0.5 * fs
								cutoff = 30
								normal_cutoff = cutoff / nyq

								b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
								out = scipy.signal.lfilter(b, a, out, axis = 0)*fs

							else:
								out = np.asarray(data_output)

							if in_port.getName() == "/processing/xsens/PoseQuaternion:i":

								if 'Position' in out_port.getName() or 'Velocity' in out_port.getName() or 'Acceleration' in out_port.getName():
									temp_data_out = np.zeros((len(data_output), 69))
									for k in range(0,23):
										temp_data_out[:,k*3:k*3+3] = out[:,k*7:k*7+3]

									out = temp_data_out

								else:
									temp_data_out = np.zeros((len(data_output), 92))
									for k in range(0,23):
										temp_data_out[:,k*4:k*4+4] = out[:,k*7+3:k*7+7]

									out = temp_data_out
						
							# Compute the average signals on the sliding window
							if(len(np.shape(out)) == 1):
								out = np.expand_dims(out, axis=1)

							output = np.mean(np.asarray(out[:,id_items*dimension:id_items*dimension+dimension]), axis = 0)
							# output = np.median(np.asarray(out[:,id_items:id_items+1]), axis = 0)

							if(out_port.getName() == '/processing/xsens/CoMVelocityNorm:o'):
								output = output[0:2]

							if(self.norm_list[id_ouput]):
								output = [np.linalg.norm(output)]
								dimension = 1

							if(self.dist_com_list[id_ouput]):
								if(len(self.current_com) == 0):
									continue

								output = np.sqrt(
									# (output[0] - self.current_com[0]) * (output[0] - self.current_com[0]) +
									# (output[1] - self.current_com[1]) * (output[1] - self.current_com[1]) +
									(output[2] - self.current_com[2]) * (output[2] - self.current_com[2]))
								output = np.expand_dims(output, axis=1)

							if(self.normalize[id_ouput][0] == 1):
								if(self.count == 0):
									self.median_normalize[id_ouput] = []

								
								self.count += 1
								self.median_normalize[id_ouput].append(output[0])

								if(self.count >=  1000):
									del self.median_normalize[id_ouput][0]
						
								# output = output - np.median(self.median_normalize)
								# output = (output - np.median(self.median_normalize[id_ouput]))/np.median(self.median_normalize[id_ouput])
								output = (output - np.median(self.init_com[id_items]))/np.median(self.init_com[id_items])

							# Send data to the ouput port
							dimension = np.shape(output)[0]
							b_out = out_port.prepare()
							b_out.clear()
							b_out.addInt(dimension)
							for dim in range(dimension):
								b_out.addDouble(output[dim])
							out_port.write()
					
					if(self.flag_timer == 0):
						self.flag_timer = 1	
					else:
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


class CallBackInitCom(yarp.BottleCallback):
	def __init__(self):
		yarp.BottleCallback.__init__(self)
		self.port = yarp.Port()
		self.flag_init = 0

	def onRead(self, bot, *args, **kwargs):
		data = bot.toString().split(' ')
		self.flag_init = int(data[0])
		return data

	def get_data(self):
		is_init = 0
		if(self.flag_init == 1):
			is_init = 1
			self.flag_init = 0
		return is_init



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

#!/usr/bin/python3

import yarp
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import scipy.spatial
from scipy.spatial.transform import Rotation as R
import time
from copy import deepcopy

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
		self.last_orientation_data = []

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


		# the orientation port must always be created
		self.input_port.append(yarp.BufferedPortBottle())
		self.input_port[nb_active_port].open("/processing/xsens/AngularSegmentKinematics:i")			
		self.list_name_ports.append("/xsens/AngularSegmentKinematics")
		self.cback.append(CallbackData(size_buffer))
		self.input_port[nb_active_port].useCallback(self.cback[nb_active_port])
		self.buffer.append([])
		nb_active_port += 1


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
						self.buffer.append([])
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
							if item_carac[1] in ['x', 'q0']:
								dim_item = 0
							elif item_carac[1] in ['y', 'q1']:
								dim_item = 1
							elif item_carac[1] in ['z', 'q2']:
								dim_item = 2
							elif item_carac[1] in ['q3']:
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
						dimension_orientation = int(rf.findGroup(related_items).find('dimension_orientation').toString())

						if signal == "orientation":
							id_item = id_item*dimension_orientation + dim_item
						else:
							id_item = id_item*dimension + dim_item
						dimension = 1

						nb_output_port += 1
						self.related_port.append([input_port_name, id_item, dimension])

		self.clock = yarp.Time.now()

		print(self.list_name_ports)

		return True

# when you close the program with CTRL+C
	def close(self):

		print('*** Closing the ports of this module ***')

		for port in self.input_port:	
			print('DEBUG closing port input ')
			tic = time.time()
			port.close()
			toc = time.time()
			print('DEBUG closed input port in ',toc-tic)

		for port in self.output_port:
			print('closing port output ')
			tic = time.time()
			port.close()
			toc = time.time()
			print('DEBUG closed port output in ',toc-tic)
			

		print('DEBUG closing port INIT COM')
		self.port_init_com.close()

		print('DEBUG closing port HANDLER ')
		self.handlerPort.close()

		print('Finished closing ports ')

		return True


	def interruptModule(self):

		print('*** Interrupt the ports of this module ***')

		for port in self.input_port:
			port.interrupt()

		for port in self.output_port:
			port.interrupt()

		self.port_init_com.interrupt()
		self.handlerPort.interrupt()

		print('Finished interrupt ports ')

		return True

	def updateModule(self):
		received_data = self.read_input_ports()

		current_time = yarp.Time.now()

		# Slidding Window
		if((current_time - self.clock) >= self.window_size):
			initalization = self.cback_init.get_data()
			#debug
			#print('delta window size ',str(current_time - self.clock))
			self.clock = current_time


			# Get the data from each input port corresponding to the window
			for in_port, id_input in zip(self.input_port, range(len(self.input_port))):	
				#print("DEBUG input port ", in_port.getName(), id_input)
				#print("debug received data", len(self.buffer), len(received_data))
				if(received_data[id_input] == 0 and len(self.buffer[id_input]) > 0):
					del self.buffer[id_input][0]

				if(len(self.buffer[id_input]) > 0):
					length = int(len(self.buffer[id_input])/2) #pour superposer les fenetres je garde la moitié des données
					
					data_output = self.buffer[id_input]

					if(in_port.getName() == '/processing/xsens/COM:i'):
						self.current_com = np.mean(np.asarray(data_output), axis = 0)
						#DEBUG
						#print(' DEBUG current COM  ',str(self.current_com))
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

							if in_port.getName() == "/processing/xsens/LinearSegmentKinematics:i":
								# 23 segments
								# 3 dimension for every segment 
								# to debug, you must check the Xsens MVN streamer documentation [MVN real time 
								# network streaming protocol specifications], section 2.7.2 Linear Segment Kinematics:
								#
								#  (segment ID, coord X segment, Y, Z, Vx, Vy, Vz, Ax, Ay, Az)
								#
								# attention, yarp ports add an extra number to the bottle, so segment ID is at index 1,
								# coord X is at index 2.
								# the numbering of the segments is fund in the MVN User Manual, page 105:
								#  1 = pelvis (index 0 in the numpy matrix temp_data_out)
								#  2 = L5 (index 1)
								#  3 = L3
								#  4 = T12
								#  5 = T8
								#  6 = Neck
								#  7 = Head
								#  8 = Right Shoulder
								#  9 = Right Upper Arm
								# 10 = Right Forearm
								# 11 = Right Hand
								# 12 - 15 = Left Shoulder / Upper Arm / Forearm / Hand
								# 16 - 19 = Right Upper Leg / LOwer Leg / Foot / Toe
								# 20 - 23 = Left Upper Leg / LOwer Leg / Foot / Toe 


								temp_data_out = np.zeros((len(data_output), 69))
								if 'Position' in out_port.getName():
									for k in range(0,23):   # from 0 (pelvis) to 22 (left toe)
										temp_data_out[:,k*3:k*3+3] = out[:,k*10+2:k*10+5]  #takes values 2, 3, 4 only!

									# DEBUG 17-09-2020
									# in Data_base.py there's a processing of position data that was enables during the 
									# binary dataset creation. the positions are all transformed to be wrt the pelvis.
									# pelvis is the "zero" for the cartesian coordinates
									# we need to do the same

									for t in range(len(temp_data_out)):
										abs_data = deepcopy(temp_data_out[t])
										o_data = self.last_orientation_data

										# get pevis orientation
										id_pelvis = 0
										q0 = o_data[id_pelvis*4]
										q1 = o_data[id_pelvis*4 + 1]
										q2 = o_data[id_pelvis*4 + 2]
										q3 = o_data[id_pelvis*4 + 3]

										# compute pelvis orientation matrix using quaternion orinetation matrix 
										# cf. xsens manual page 123, section 23.2.2
										R = np.array([[q0*q0 + q1*q1 - q2*q2 - q3*q3, 2*q1*q2 - 2*q0*q3, 2*q1*q3 + 2*q0*q2],
											[2*q1*q2 + 2*q0*q3, q0*q0 - q1*q1 + q2*q2 - q3*q3, 2*q2*q3 - 2*q0*q1],
											[2*q1*q3 - 2*q0*q2, 2*q2*q3 + 2*q0*q1, q0*q0 - q1*q1 - q2*q2 + q3*q3]])

										# TO FIX LATER: this should be from 1 to 23 -> range(1,23)
										for i in range(0, 23):
											# compute distance position wrt the pelvis position (pelvis is zero)
											temp_data_out[t,i*3:i*3+3] = abs_data[i*3:i*3+3] - abs_data[0:3]
											temp_data_out[t,i*3:i*3+3] = R@temp_data_out[t,i*3:i*3+3]



								elif 'Velocity' in out_port.getName():
									for k in range(0,23):
										temp_data_out[:,k*3:k*3+3] = out[:,k*10+5:k*10+8] #takes values 5, 6, 7 only!

								elif 'Acceleration' in out_port.getName():
									for k in range(0,23):
										temp_data_out[:,k*3:k*3+3] = out[:,k*10+8:k*10+11]

								out = temp_data_out




							elif in_port.getName() == "/processing/xsens/AngularSegmentKinematics:i":

								# we have 4 here because it's quaternions 


								if 'Orientation' in out_port.getName():
									temp_data_out = np.zeros((len(data_output), 92))
									for k in range(0,23):
										# ATTENTION: in YARP Xsens streamer the orientation signals are converted 
										# in deg using rad2deg even if they are quaternions
										# while the original xsens signal is simply quaternions:
										# this means that the training is done with quaternions but in the online
										# recognition demo we need to convert these signals to match
										# the input used in the training 
										#temp_data_out[:,k*4:k*4+4] = np.deg2rad(out[:,k*11+2:k*11+6])
										# 
										# ATTENTION: this is only if YARP xsens streamer does not make
										# a rad2deg (debugged version by Serena)
										temp_data_out[:,k*4:k*4+4] = out[:,k*11+2:k*11+6]

								elif 'AngularVelocity' in out_port.getName():
									temp_data_out = np.zeros((len(data_output), 69))
									for k in range(0,23):
										# attention: 
										# AndyDataset (used for training) uses radians
										# Xsens MVN streamer streams in radians
										# but our YARP xsens streamer converts in degrees (makes rad2deg)
										temp_data_out[:,k*3:k*3+3] = np.deg2rad(out[:,k*11+6:k*11+9])
										
								elif 'AngularAcceleration' in out_port.getName():
									temp_data_out = np.zeros((len(data_output), 69))
									for k in range(0,23):
										# attention: 
										# AndyDataset (used for training) uses radians
										# YARP Xsens MVN streamer streams in degrees (makes rad2deg)
										temp_data_out[:,k*3:k*3+3] = np.deg2rad(out[:,k*11+9:k*11+12])
										# temp_data_out[:,k*3:k*3+3] = out[:,k*11+9:k*11+12]

								out = temp_data_out

						
							# Compute the average signals on the sliding window
							if(len(np.shape(out)) == 1):
								out = np.expand_dims(out, axis=1)

							output = np.mean(np.asarray(out[:,id_items*dimension:id_items*dimension+dimension]), axis = 0)

							# NOT USED ANYMORE
							# if(out_port.getName() == '/processing/xsens/CoMVelocityNorm:o'):
							# 	# norm in x-y only. not used anymore
							# 	output = output[0:2]

							# this is for the norm of an entire vector; for example for velocity
							if(self.norm_list[id_ouput]):
								output = [np.linalg.norm(output)]
								dimension = 1


							# NOT USED ANYMORE
							# distance of COM - not used
							# if(self.dist_com_list[id_ouput]):
							# 	if(len(self.current_com) == 0):
							# 		continue
							# 	output = np.sqrt(
							# 		# (output[0] - self.current_com[0]) * (output[0] - self.current_com[0]) +
							# 		# (output[1] - self.current_com[1]) * (output[1] - self.current_com[1]) +
							# 		(output[2] - self.current_com[2]) * (output[2] - self.current_com[2]))
							# 	output = np.expand_dims(output, axis=1)


							# NORMALIZING THE VALUES OF THE SIGNAL BETWEEN 0,-1 (approx)
							# used for the COM
							# it takes the first 1000 values to consider that this is the zero of the human COM
							# because it is supposed to start upright (the standing pose of xsens for calibration)
							# important: it is like that in the binary dataset 
							if(self.normalize[id_ouput][0] == 1):
								if(self.count == 0):
									self.median_normalize[id_ouput] = []
								self.count += 1
								self.median_normalize[id_ouput].append(output[0])
								if(self.count >=  1000):
									del self.median_normalize[id_ouput][0]

								#print('DEBUG output ',output)
								output = (output - np.median(self.median_normalize[id_ouput]))/np.median(self.median_normalize[id_ouput])
								#print('DEBUG output normalized ',output)

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

			# 17-09-2020
			# this is dirty, but we need to save always the orientation values to process the position
			# data wrt to the pelvis - check update() function
			if(len(data)>0):

				#print("DEBUG data from ori ", np.shape(data))
				#print("DEBUG DATA ", data)

				if port.getName() == "/processing/xsens/AngularSegmentKinematics:i":
					temp_data_orientation = np.zeros(92) #np.zeros((len(data), 92))
					for k in range(0,23):
						temp_data_orientation[k*4:k*4+4] = data[0][k*11+2:k*11+6]
					self.last_orientation_data = temp_data_orientation

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
	
	# testing.. comment if it doesnt zork and restore the part below
	mod_sensor.runModule(rf)

	# while(True):
	# 	try:
	# 		yarp.Time.delay(mod_sensor.getPeriod())
	# 		mod_sensor.updateModule()
			
	# 	except KeyboardInterrupt:
	# 		print('*** I detected a CTRL+C ... ')
	# 		#mod_sensor.close()
	# 		break

	print('Exited after CTRL+C')

	yarp.Network.fini();

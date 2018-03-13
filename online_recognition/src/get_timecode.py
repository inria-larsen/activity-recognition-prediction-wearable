#!/usr/bin/python3

import yarp
import sys
import matplotlib.pyplot as plt
import numpy as np

class GetTimeCode():


	def __init__(self):
		self.input_port = yarp.BufferedPortBottle()
		self.input_port.open('/timecode:i')

		self.output_port = yarp.BufferedPortBottle()
		self.output_port.open('/timecode:o')


	def update(self):
		b_in = self.input_port.read()
		timecode = b_in.toString().split(' ')
		b_out = self.output_port.prepare()
		b_out.clear()
		b_out.addDouble(float(timecode[0]))
		self.output_port.write()
		return

	def close(self):
		self.input_port.close()

if __name__=="__main__":
	yarp.Network.init()
	rf = yarp.ResourceFinder()
	rf.configure(sys.argv)
	
	timecode_module = GetTimeCode()

	while(True):
		try:
			timecode_module.update()
		except KeyboardInterrupt:
			timecode_module.close()
			break

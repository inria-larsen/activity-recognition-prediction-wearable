#!/usr/bin/python3

import yarp
import time
import sys
import configparser

class RFModule(yarp.RFModule):
	def __init__(self, name_port):
		yarp.RFModule.__init__(self)
		self.name_port = name_port

	def configure(self, rf):
		self.name_port = self.name_port
		self.port = yarp.BufferedPortBottle()
		self.port.open(self.name_port)

		moduleName = rf.check("name", yarp.Value(self.name_port)).asString()
		self.setName(moduleName)
		print(moduleName)

		return True 

	def close(self):
		print("Closing ports")
		self.port.close()

	def interruptModule(self):
		self.port.interrupt()
		return True

	def getPeriod(self):
		return 0.001

	def updateModule(self):
		b_in = self.port.read()		
		b_out = self.port.prepare()
		b_out.clear()
		b_out.append(b_in)
		self.port.write()
		return True

if __name__=="__main__":
	name = sys.argv[1]

	rf = yarp.ResourceFinder()

	mod = RFModule(name)
	mod.configure(rf)
	mod.runModule(rf)



import yarp
import sys
import numpy as np


class DetectionContact(object):
	def __init__(self):
		self.input_port = '/processing/eglove/data/Forces:o'

		self.port = yarp.BufferedPortBottle()
		self.port.open('/DetectionContact')

		yarp.Network.connect(self.input_port, self.port.getName())

	def update(self):
		b_in = self.port.read()
		data = b_in.toString().split(' ')

		del data[0]

		glove_forces = float(data[0])

		contact = 0
		if(glove_forces > 10):
			contact = 1

		b_out = self.port.prepare()
		b_out.clear()
		b_out.addInt(contact)
		self.port.write()



def main():
	yarp.Network.init()
	rf = yarp.ResourceFinder()
	rf.setDefaultContext("online_recognition")
	rf.setDefaultConfigFile("eglove_only.ini")
	rf.configure(sys.argv)

	contact_module = DetectionContact()

	while(True):
		try:
			contact_module.update()
		except KeyboardInterrupt:
			break



if __name__ == '__main__':
    main()
		
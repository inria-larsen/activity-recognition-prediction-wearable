#!/usr/bin/python3

import yarp
import sys
import matplotlib.pyplot as plt
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np


class RealTimePlotModule():
	"""
	This module plots a bar chart with the probability distribution on the states.
	Usage
	python plot_probabilities.py
	Input port: /processing/NamePort:o
	"""
	def __init__(self, name_port, size_window):

		self.input_port = name_port
		self.port = yarp.BufferedPortBottle()
		self.port.open('/plot' + name_port)

		self.plotWindow = pg.GraphicsWindow(title=name_port)
		self.plotData = self.plotWindow.addPlot(title=name_port)

		self.buffer = [[]]
		self.list_curve = []

		self.size_window = size_window

		print(name_port)

		yarp.Network.connect(name_port, self.port.getName())

		self.flag_init = 0


	def update_plot(self):
		b_in = self.port.read()
		data = b_in.toString().split(' ')

		dimension = int(data[0])

		del data[0]
		if(self.flag_init == 0):
			for dim in range(dimension):
				self.list_curve.append(self.plotData.plot(pen=(dim,dimension)))
			self.flag_init = 1

		value = list(map(float, data))

		for dim in range(dimension):
			if(len(self.buffer) <= dim):
				self.buffer.append([])

			self.buffer[dim].append(value[dim])
			if(len(self.buffer[dim]) > self.size_window):
				del self.buffer[dim][0]

		for dim in range(dimension):
			self.list_curve[dim].setData(self.buffer[dim])

		QtGui.QApplication.processEvents()
		return

	def close(self):
		yarp.Network.disconnect(self.input_port, self.port.getName())
		self.port.close()


if __name__=="__main__":
	yarp.Network.init()
	rf = yarp.ResourceFinder()
	rf.configure(sys.argv)
	
	name_port = rf.find("name_port").toString()
	size_window = rf.find("size_window").toString()

	if(len(size_window) == 0):
		size_window = 3000
	else:
		size_window = int(size_window)

	fig = RealTimePlotModule(name_port, size_window=size_window)

	while(True):
		try:
			fig.update_plot()
			i = 0
		except KeyboardInterrupt:
			fig.close()
			break


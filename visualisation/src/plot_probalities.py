#!/usr/bin/python3

import yarp
import sys
import matplotlib.pyplot as plt
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np


class RealTimeBarPlotModule():
	"""
	This module plots a bar chart with the probability distribution on the states.
	Usage
	python plot_probabilities.py
	Input port: /activity_recognition/probabilities
	"""
	def __init__(self, rf):
		path_model = rf.find('path_model').toString()

		self.input_port = '/activity_recognition/probabilities:o'
		self.port = yarp.BufferedPortBottle()
		self.port.open('/plot/probabilities')

		self.plotWindow = pg.GraphicsWindow(title='Probabilities')
		self.plotData = self.plotWindow.addPlot(title='Probabilities')

		self.buffer = [[]]
		self.list_curve = []

		yarp.Network.connect(self.input_port, self.port.getName())

		self.flag_init = 0


	def update_plot(self):
		b_in = self.port.read()
		data = b_in.toString().split(' ')

		states = []
		prob = []

		for i in range(0, len(data), 2):
			states.append(data[i])
			prob.append(float(data[i + 1]))

		x = np.arange(len(states))

		self.plotData.clear()

		bg1 = pg.BarGraphItem(x=x, height=prob, width=0.6, brush='r')

		self.plotData.addItem(bg1)
		label_states = dict(enumerate(states))
		self.plotData.getAxis('bottom').setTicks([label_states.items()])
		QtGui.QApplication.processEvents()
		return

	def close(self):
		yarp.Network.disconnect(self.input_port, self.port.getName())
		self.port.close()


if __name__=="__main__":
	yarp.Network.init()
	rf = yarp.ResourceFinder()
	rf.setDefaultContext("online_recognition")
	rf.setDefaultConfigFile("default.ini")
	rf.configure(sys.argv)

	fig = RealTimeBarPlotModule(rf)
	
	while(True):
		try:
			fig.update_plot()
		except KeyboardInterrupt:
			fig.close()
			break


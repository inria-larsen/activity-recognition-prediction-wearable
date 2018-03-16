import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import andy_reco
import random

import yarp

ListActivities = {
    'walking_upright'
    'standing_upright'
    'standing_bent_forward'
    'standing_strongly_bent_forward'
    'standing_overhead_work_elbow_at_above_shoulder'
    'standing_overhead_work_hands_above_head'
    'kneeling_upright'
    'kneeling_bent'
    'kneeling_elbow_at_above_shoulder'
}

class Cockpit(QtWidgets.QMainWindow, andy_reco.Ui_MainWindow):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        # we update the probability at 100 Hz
        self.__timer = QTimer()
        self.__timer.timeout.connect(self.update_probabilities)
        self.__timer.start(10)# every 10 ms

        # Prepare yarp ports
        self.activity_input = '/activity_recognition/probabilities:o'
        self.port_activity = yarp.BufferedPortBottle()
        self.port_activity.open('/gui/probabilities')

        yarp.Network.connect(self.activity_input, self.port_activity.getName())

        self.contact_input = '/DetectionContact'
        self.port_contact = yarp.BufferedPortBottle()
        self.port_contact.open('/gui/contact')

        yarp.Network.connect(self.contact_input, self.port_contact.getName())

        self.flag_init = 0


    def update_probabilities(self):
        b_in = self.port_activity.read()
        data = b_in.toString().split(' ')

        score = []

        for i in range(1, len(data), 2):
            score.append(float(data[i]))
        
        max_value = max(score)
        id_max = score.index(max_value)
        current_state = data[id_max*2]

        self.current_activity_image.setPixmap(QPixmap('/home/amalaise/Documents/These/code/activity-recognition-prediction-wearable/visualisation/app/figs/' + current_state + '.png'))
        self.current_activity_image.setScaledContents( True )
        self.standing_strongly_bent.setValue(float(data[data.index('standing_strongly_bent_forward') + 1]) * 100)
        self.standing_bent_forward.setValue(float(data[data.index('standing_bent_forward') + 1]) * 100)
        self.walking_upright.setValue(float(data[data.index('walking_upright') + 1]) * 100)
        self.standing_upright.setValue(float(data[data.index('standing_upright') + 1]) * 100)
        self.standing_overhead_elbow.setValue(float(data[data.index('standing_overhead_work_elbow_at_above_shoulder') + 1]) * 100)
        self.standing_overhead_hands.setValue(float(data[data.index('standing_overhead_work_hands_above_head') + 1]) * 100)
        self.kneeling_upright.setValue(float(data[data.index('kneeling_upright') + 1]) * 100)
        self.kneeling_bent.setValue(float(data[data.index('kneeling_bent') + 1]) * 100)
        self.kneeling_elbow.setValue(0)

        b_in2 = self.port_contact.read()
        data = b_in2.toString().split(' ')

        if(int(data[0])):
            self.object_in_hand.setPixmap(QPixmap('/home/amalaise/Documents/These/code/activity-recognition-prediction-wearable/visualisation/app/figs/objYes.png'))
        else:
            self.object_in_hand.setPixmap(QPixmap('/home/amalaise/Documents/These/code/activity-recognition-prediction-wearable/visualisation/app/figs/objNo.png'))
        
        self.object_in_hand.setScaledContents( True )

        self.object_yes.setValue(float(data[0]) * 100)
        self.object_no.setValue(100 - float(data[0]) * 100)


def main():
    yarp.Network.init()
    rf = yarp.ResourceFinder()
    rf.setDefaultContext("online_recognition")
    rf.setDefaultConfigFile("default.ini")
    rf.configure(sys.argv)

    app = QtWidgets.QApplication(sys.argv)
    form = Cockpit()
    form.show()
    app.exec_()


if __name__ == '__main__':
    main()

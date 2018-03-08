import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import andy_reco
import random

class Cockpit(QtWidgets.QMainWindow, andy_reco.Ui_MainWindow):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        # we update the probability at 100 Hz
        self.__timer = QTimer()
        self.__timer.timeout.connect(self.update_probabilities)
        self.__timer.start(10)# every 10 ms

    def update_probabilities(self):
        self.standing_strongly_bent.setValue(random.randint(0,100))
        self.standing_bent_forward.setValue(random.randint(0,100))
        self.walking_upright.setValue(random.randint(0,100))
        self.standing_upright.setValue(random.randint(0,100))
        self.standing_overhead_elbow.setValue(random.randint(0,100))
        self.standing_overhead_hands.setValue(random.randint(0,100))
        self.kneeling_upright.setValue(random.randint(0,100))
        self.kneeling_bent.setValue(random.randint(0,100))
        self.kneeling_elbow.setValue(random.randint(0,100))
        self.object_yes.setValue(random.randint(0,100))
        self.object_no.setValue(random.randint(0,100))

def main():
    app = QtWidgets.QApplication(sys.argv)
    form = Cockpit()
    form.show()
    app.exec_()


if __name__ == '__main__':
    main()

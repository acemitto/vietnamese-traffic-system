from time import localtime, asctime
from datetime import datetime
from PyQt5 import QtWidgets
from PyQt5.uic import loadUi

from Database import KEYS
from DetailLogWindow import DetailLogWindow


class ViolationItem (QtWidgets.QListWidget):
    def __init__ (self, parent = None):
        super(ViolationItem, self).__init__(parent)
        loadUi("/Users/acemitto/Desktop/STUDY/ACE/CAPSTONE/vietnamese-traffic-system/Source/UI/ViolationItem.ui", self)
        self.details_button.clicked.connect(self.showDetails)
        self.data = {}

    def setData(self, data):
        self.data = data
        self.setCarId(data[KEYS.CARID])
        self.setTime(data[KEYS.TIME])
        self.setCarImage(data[KEYS.CARIMAGE])

    def showDetails(self):
        window = DetailLogWindow(self.data, self)
        window.show()

    def setCarId(self, id):
        self.carid.setText(str(id))

    def setTime(self, time):
        print(time)
        self.time.setText(time)

    def setCarImage(self, pixmap):
        self.carimage.setPixmap(pixmap)
        self.carimage.show()


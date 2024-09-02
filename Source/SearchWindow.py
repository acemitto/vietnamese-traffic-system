from PyQt5 import QtWidgets
from PyQt5.QtCore import QStringListModel
from PyQt5.QtWidgets import QMainWindow, QCompleter
from PyQt5.uic import loadUi

from Database import Database
from ViolationItem import ViolationItem


class SearchWindow(QMainWindow):
    def __init__(self, search_result, parent=None):
        super(SearchWindow, self).__init__(parent)
        loadUi("/Users/acemitto/Desktop/STUDY/ACE/CAPSTONE/vietnamese-traffic-system/Source/UI/Search.ui", self)

        self.search_result = search_result

        self.type.addItems(["None"])
        self.type.addItems(Database.getInstance().getCarTypesList())

        completer = QCompleter()
        self.substring.setCompleter(completer)
        model = QStringListModel()
        completer.setModel(model)
        licenseList = Database.getInstance().getLicenseList()
        model.setStringList(licenseList)

        self.search_button.clicked.connect(self.search)

        cams = Database.getInstance().getCamList('All')
        self.camera.clear()
        self.camera.addItems(["None"])
        self.camera.addItems(cam_name for cam_name, cam_location, cam_feed in cams)
        self.camera.setCurrentIndex(0)

    def search(self):
        cam = None if self.camera.currentText() == "None" else self.camera.currentText()
        type = None if self.type.currentText() == "None" else self.type.currentText()
        license = None if self.substring.text() == "" else self.substring.text()
        time = None if self.use_time.isChecked() is False else (self.from_time.dateTime().toMSecsSinceEpoch(), self.to_time.dateTime().toMSecsSinceEpoch())
        rows = Database.getInstance().search(cam=cam, type=type, license=license, time=time)
        for row in rows:
            print(row)
            listWidget = ViolationItem()
            listWidget.setData(row)
            listWidgetItem = QtWidgets.QListWidgetItem(self.search_result)
            listWidgetItem.setSizeHint(listWidget.sizeHint())
            self.search_result.addItem(listWidgetItem)
            self.search_result.setItemWidget(listWidgetItem, listWidget)
        self.destroy()

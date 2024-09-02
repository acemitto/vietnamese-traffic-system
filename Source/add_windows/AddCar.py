from PyQt5.QtWidgets import QFileDialog
from PyQt5 import QtCore
from Database import Database
from add_windows.AddMainWindow import AddMainWindow


class AddCar(AddMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent, "/Users/acemitto/Desktop/STUDY/ACE/CAPSTONE/vietnamese-traffic-system/Source/Traffic-Rule-Violation-Detection-System-master/app/UI/AddCar.ui")
        self.license_browse.clicked.connect(lambda: self.getFile(self.license_img))
        self.car_browse.clicked.connect(lambda: self.getFile(self.car_img))

        carTypes = Database.getInstance().getCarTypesList()
        self.car_types.clear()
        self.car_types.addItems(name for name in carTypes)
        self.car_types.setCurrentIndex(0)
        

    def addToDatabase(self):
        car_type = str(self.car_types.currentText())
        lic_num = str(self.license_num.text())
        lic_img = str(self.license_img.text())
        car_img = str(self.car_img.text())
        Database.getInstance().insertIntoCars(car_type, lic_img, lic_num, car_img)
        self.destroy()

    def getFile(self, lineEdit):
        lineEdit.setText(QFileDialog.getOpenFileName()[0])

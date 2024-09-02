import time
import cv2
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QStatusBar, QListWidget, QAction, qApp, QMenu
from PyQt5.uic import loadUi
import numpy as np
from Archive import ArchiveWindow
from Database import Database
from processor.MainProcessor import MainProcessor
from processor.TrafficProcessor import TrafficProcessor
from ViolationItem import ViolationItem
from add_windows.AddCamera import AddCamera
from add_windows.AddCar import AddCar
from add_windows.AddRule import AddRule
from add_windows.AddViolation import AddViolation

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi("/Users/acemitto/Desktop/STUDY/ACE/CAPSTONE/vietnamese-traffic-system/Source/UI/MainWindow.ui", self)

        self.live_preview.setScaledContents(True)
        from PyQt5.QtWidgets import QSizePolicy
        self.live_preview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        self.cam_clear_gaurd = False

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Welcome!!!")

        self.search_button.clicked.connect(self.search)
        self.clear_button.clicked.connect(self.clear)
        self.refresh_button.clicked.connect(self.refresh)

        self.database = Database.getInstance()

        cam_groups = self.database.getCamGroupList()
        self.camera_group.clear()
        self.camera_group.addItems(group_name for group_name in cam_groups)
        self.camera_group.setCurrentIndex(0)
        self.camera_group.currentIndexChanged.connect(self.camGroupChanged)
        cams = self.database.getCamList(self.camera_group.currentText())
        self.cam_selector.clear()
        self.cam_selector.addItems(cam_name for cam_name, cam_location, cam_feed in cams)
        self.cam_selector.setCurrentIndex(0)
        self.cam_selector.currentIndexChanged.connect(self.camChanged)
        self.processor = MainProcessor(self.cam_selector.currentText())

        self.log_tabwidget.clear()
        self.violation_list = QListWidget(self)
        self.search_result = QListWidget(self)
        self.log_tabwidget.addTab(self.violation_list, "Violations")
        self.log_tabwidget.addTab(self.search_result, "Search Result")

        self.feed = None
        self.vs = None
        self.updateCamInfo()

        self.updateLog()

        self.initMenu()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_image)
        self.timer.start(50)

    #     # trafficLightTimer = QTimer(self)
    #     # trafficLightTimer.timeout.connect(self.toggleLight)
    #     # trafficLightTimer.start(5000)

    # def toggleLight(self):
    #     self.processor.setLight('Green' if self.processor.getLight() == 'Red' else 'Red')

    def initMenu(self):
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')

        # File menu

        ## add record manually
        addRec = QMenu("Add Record", self)

        act = QAction('Add Car', self)
        act.setStatusTip('Add Car Manually')
        act.triggered.connect(self.addCar)
        addRec.addAction(act)

        act = QAction('Add Rule', self)
        act.setStatusTip('Add Rule Manually')
        act.triggered.connect(self.addRule)
        addRec.addAction(act)

        act = QAction('Add Violation', self)
        act.setStatusTip('Add Violation Manually')
        act.triggered.connect(self.addViolation)
        addRec.addAction(act)

        act = QAction('Add Camera', self)
        act.setStatusTip('Add Camera Manually')
        act.triggered.connect(self.addCamera)
        addRec.addAction(act)

        fileMenu.addMenu(addRec)

        # check archive record ( Create window and add button to restore them)
        act = QAction('&Archives', self)
        act.setStatusTip('Show Archived Records')
        act.triggered.connect(self.showArch)
        fileMenu.addAction(act)

        ## Add Exit
        fileMenu.addSeparator()
        act = QAction('&Exit', self)
        act.setShortcut('Ctrl+Q')
        act.setStatusTip('Exit application')
        act.triggered.connect(qApp.quit)
        fileMenu.addAction(act)

    # def keyReleaseEvent(self, event):
    #     if event.key() == QtCore.Qt.Key_G:
    #         self.processor.setLight("Green")
    #     elif event.key() == QtCore.Qt.Key_R:
    #         self.processor.setLight("Red")
    #     elif event.key() == QtCore.Qt.Key_S:
    #         self.toggleLight()

    def addCamera(self):
        addWin = AddCamera(parent=self)
        addWin.show()

    def addCar(self):
        addWin = AddCar(parent=self)
        addWin.show()

    def addViolation(self):
        pass
        addWin = AddViolation(parent=self)
        addWin.show()

    def addRule(self):
        addWin = AddRule(parent=self)
        addWin.show()

    def showArch(self):
        addWin = ArchiveWindow(parent=self)
        addWin.show()

    def updateSearch(self):
        pass

    # def update_image(self):
    #     _, frame = self.vs.read()

    #     packet = self.processor.getProcessedImage(frame)
    #     cars_violated = packet['list_of_cars']  # list of cropped images of violated cars
    #     if len(cars_violated) > 0:
    #         for c in cars_violated:
    #             carId = self.database.getMaxCarId() + 1
    #             car_img = 'car_' + str(carId) + '.png'
    #             cv2.imwrite('car_images/' + car_img, c)
    #             self.database.insertIntoCars(car_id=carId, car_img=car_img)

    #             self.database.insertIntoViolations(camera=self.cam_selector.currentText(), car=carId, rule='1',
    #                                                time=time.time())

    #         self.updateLog()

    #     qimg = self.toQImage(packet['frame'])
    #     self.live_preview.setPixmap(QPixmap.fromImage(qimg))

    def update_image(self):
        _, frame = self.vs.read()

        packet = self.processor.getProcessedImage()
        

        qimg = self.toQImage(packet)
        self.live_preview.setPixmap(QPixmap.fromImage(qimg))


    def updateCamInfo(self):
        count, cam_location, self.feed = self.database.getCamDetails(self.cam_selector.currentText())
        self.feed = '/Users/acemitto/Desktop/STUDY/ACE/CAPSTONE/vietnamese-traffic-system/Source/videos/' + self.feed
        if self.cam_selector.currentText() == 'cam_01':
            pl_lane = [np.array([
                [520, 327],[2027, 327],[2960, 1400],[-451, 1400]
            ])]
            pl = [
                np.array([
                    [520, 327],[862, 327],[380, 1400],[-451, 1400]
                ]),
                np.array([
                    [880, 327],[1274, 327],[1239, 1400],[426, 1400]
                ]),
                np.array([
                    [1292, 327],[1672, 327],[2055, 1400],[1274, 1400]
                ]),
                np.array([
                    [1693, 327],[2027, 327],[2960, 1400],[2096, 1400]
                ])
            ]
            yl = 630
            yj = 765
        if self.cam_selector.currentText() == 'cam_02':
            pl_lane = [np.array([
                [554, 270],[2027, 270],[2600, 1440],[-73, 1440]
            ])]
            pl = [
                np.array([
                    [554, 270],[1030, 270],[758, 1440],[-73, 1440]
                ]),
                np.array([
                    [1054, 270],[1520, 270],[1680, 1440],[830, 1440]
                ]),
                np.array([
                    [1547, 270],[2027, 270],[2600, 1440],[1725, 1440]
                ])
            ]
            yl = 628
            yj = 1032
        if self.cam_selector.currentText() == 'cam_03':
            pl_lane = [np.array([
                [554, 270],[2027, 270],[2600, 1440],[-73, 1440]
            ])]
            pl = [
                np.array([
                    [554, 270],[1030, 270],[758, 1440],[-73, 1440]
                ]),
                np.array([
                    [1054, 270],[1520, 270],[1680, 1440],[830, 1440]
                ]),
                np.array([
                    [1547, 270],[2027, 270],[2600, 1440],[1725, 1440]
                ])
            ]
            yl = 628
            yj = 1032
        if self.cam_selector.currentText() == 'cam_04':
            pl_lane = [np.array([
                [981, 32],[1640, 32],[1929, 1440],[-275, 1440]
            ])]
            pl = [
                np.array([
                    [1140, 32],[1380, 32],[1128, 1440],[359, 1440]
                ]),
                np.array([
                    [1390, 32],[1640, 32],[1929, 1440],[1155, 1440]
                ])
            ]
            yl = 690
            yj = 900
        if self.cam_selector.currentText() == 'cam_1255':
            pl_lane = [np.array([
                [263, 159],[1280, 159],[1280, 720],[-266, 720]
            ])]
            pl = [
                np.array([
                    [263, 159],[450, 159],[117, 720],[-266, 720]
                ]),
                np.array([
                    [459, 159],[650, 159],[549, 720],[137, 720]
                ]),
                np.array([
                    [661, 159],[868, 159],[1016, 720],[571, 720]
                ]),
                np.array([
                    [879, 159],[1280, 159],[1280, 720],[1037, 720]
                ])
            ]
            yl = 207
            yj = 337
            yz=0
        if self.cam_selector.currentText() == 'cam_4659':
            pl_lane = [np.array([
                [854, 550],[1518, 550],[2059, 1440],[210, 1440]
            ])]
            pl = [
                np.array([
                    [854, 550],[1076, 550],[840, 1440],[210, 1440]
                ]),
                np.array([
                    [1098, 550],[1308, 550],[1473, 1440],[860, 1440]
                ]),
                np.array([
                    [1319, 550],[1518, 550],[2059, 1440],[1500, 1440]
                ])
            ]
            yl = 814
            yj = 928
            yz=0
        if self.cam_selector.currentText() == 'video 4':
            pl_lane = [np.array([
                [1600, 1444],[2835, 1444],[3435, 2160],[1040, 2160]
            ])]
            pl = [
                np.array([
                    [1600, 1444],[2193, 1444],[2200, 2160],[1040, 2160]
                ]),
                np.array([
                    [2220, 1444],[2835, 1444],[3435, 2160],[2252, 2160]
                ])
            ]
            yl = 1750
            yj = 1845
            yz=0
        if self.cam_selector.currentText() == 'video 1':
            pl_lane = [np.array([
                [1823, 1570],[2343, 1570],[2455, 2160],[1475, 2160]
            ])]
            pl = [
                np.array([
                    [1823, 1570],[2087, 1570],[1964, 2160],[1475, 2160]
                ]),
                np.array([
                    [2087, 1570],[2343, 1570],[2455, 2160],[1964, 2160]
                ])
            ]
            yl = 1829
            yj = 1931
            yz= 1570
        self.processor.preCode(str(self.feed), pl_lane, pl, yl, yj)
        self.processor = MainProcessor(self.cam_selector.currentText())
        self.vs = cv2.VideoCapture(self.feed)
        self.cam_id.setText(self.cam_selector.currentText())
        self.address.setText(cam_location)
        self.total_records.setText(str(count))
        self.updateLog()

    def updateLog(self):
        self.violation_list.clear()
        rows = self.database.getViolationsFromCam(str(self.cam_selector.currentText()))
        for row in rows:
            listWidget = ViolationItem()
            listWidget.setData(row)
            listWidgetItem = QtWidgets.QListWidgetItem(self.violation_list)
            listWidgetItem.setSizeHint(listWidget.sizeHint())
            self.violation_list.addItem(listWidgetItem)
            self.violation_list.setItemWidget(listWidgetItem, listWidget)

    @QtCore.pyqtSlot()
    def refresh(self):
        self.updateCamInfo()
        self.updateLog()

    @QtCore.pyqtSlot()
    def search(self):
        from SearchWindow import SearchWindow
        searchWindow = SearchWindow(self.search_result, parent=self)
        searchWindow.show()

    @QtCore.pyqtSlot()
    def clear(self):
        qm = QtWidgets.QMessageBox
        prompt = qm.question(self, '', "Are you sure to reset all the values?", qm.Yes | qm.No)
        if prompt == qm.Yes:
            self.database.clearCamLog()
            self.updateLog()
            # self.database.deleteAllCars()
            # self.database.deleteAllViolations()
        else:
            pass

    def toQImage(self, raw_img):
        from numpy import copy
        img = copy(raw_img)
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImg = QImage(img.tobytes(), img.shape[1], img.shape[0], img.strides[0], qformat)
        outImg = outImg.rgbSwapped()
        return outImg

    @QtCore.pyqtSlot()
    def camChanged(self):
        if not self.cam_clear_gaurd:
            self.updateCamInfo()
            self.updateLog()

    @QtCore.pyqtSlot()
    def camGroupChanged(self):
        cams = self.database.getCamList(self.camera_group.currentText())
        self.cam_clear_gaurd = True
        self.cam_selector.clear()
        self.cam_selector.addItems(name for name, location, feed in cams)
        self.cam_selector.setCurrentIndex(0)
        # self.cam_selector.currentIndexChanged.connect(self.camChanged)
        self.cam_clear_gaurd = False
        self.updateCamInfo()

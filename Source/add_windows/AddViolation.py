from Database import Database
from add_windows.AddMainWindow import AddMainWindow


class AddViolation(AddMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent, "/Users/acemitto/Desktop/STUDY/ACE/CAPSTONE/vietnamese-traffic-system/Source/Traffic-Rule-Violation-Detection-System-master/app/UI/AddViolation.ui")

        cam_groups = Database.getInstance().getCamGroupList()
        self.camera.clear()
        self.camera_group.clear()
        self.camera_group.addItems(group_name for group_name in cam_groups)
        self.camera_group.setCurrentIndex(0)
        self.camera_group.currentIndexChanged.connect(self.camGroupChanged)
        cams = Database.getInstance().getCamList(self.camera_group.currentText())
        self.camera.clear()
        self.camera.addItems(cam_name for cam_name, cam_location, cam_feed in cams)
        self.camera.setCurrentIndex(0)

        carTypes = Database.getInstance().getLicenseList()
        self.car.clear()
        self.car.addItems(name for name in carTypes)
        self.car.setCurrentIndex(0)

        rules = Database.getInstance().getRulesList()
        self.rule.clear()
        self.rule.addItems(name for name in rules)
        self.rule.setCurrentIndex(0)

    def addToDatabase(self):
        camera_group = str(self.camera_group.currentText())
        camera = str(self.camera.currentText())
        license_number = str(self.car.currentText())
        rule_name = str(self.rule.currentText())
        time = self.time.dateTime().toString("yyyy-MM-dd HH:mm:ss")
        Database.getInstance().insertIntoViolations(camera_group, camera, license_number, rule_name, time)
        self.destroy()

    def camGroupChanged(self):
        cams = Database.getInstance().getCamList(self.camera_group.currentText())
        self.camera.clear()
        self.camera.addItems(cam_name for cam_name, cam_location, cam_feed in cams)
        self.camera.setCurrentIndex(0)

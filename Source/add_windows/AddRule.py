from Database import Database
from add_windows.AddMainWindow import AddMainWindow


class AddRule(AddMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent, "/Users/acemitto/Desktop/STUDY/ACE/CAPSTONE/vietnamese-traffic-system/Source/Traffic-Rule-Violation-Detection-System-master/app/UI/AddRule.ui")

    def addToDatabase(self):
        rule = str(self.rule.text())
        fine = str(self.fine.text())
        Database.getInstance().insertIntoRules(rule, fine)
        self.destroy()

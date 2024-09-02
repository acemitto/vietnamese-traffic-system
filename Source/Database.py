import sqlite3
from enum import Enum

from PyQt5.QtGui import QPixmap


class KEYS(Enum):
    LOCATION = 'location'
    CARID = 'carid'
    CARTYPE = 'cartype'
    CARIMAGE = 'carimage'
    LICENSENUMBER = 'licensenumber'
    LICENSEIMAGE = 'licenseimage'
    CAROWNER = 'carowner'
    RULENAME = 'rulename'
    RULEFINE = 'rulefine'
    TIME = 'time'
    RULEID = 'ruleid'


class Database():
    __instance = None

    @staticmethod
    def getInstance():
        if Database.__instance is None:
            Database()
        return Database.__instance

    def __init__(self):
        if Database.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            Database.__instance = self
            self.con = sqlite3.connect("/Users/acemitto/Desktop/STUDY/ACE/CAPSTONE/vietnamese-traffic-system/Source/database/traffic.db")

    # Get
    def getCamGroupList(self):
        command = "select group_name from camera_group"
        cur = self.con.cursor()
        cur.execute(command)
        rows = cur.fetchall()
        ret = [row[0] for row in rows]
        cur.close()
        return ret
    
    def getCamList(self, group):
        if str(group) != 'All':
            command = "select cam_name, cam_location, cam_feed from camera where group_id = (select group_id from camera_group where group_name = '{}')".format(str(group))
        else:
            command = "select cam_name, cam_location, cam_feed from camera"

        cur = self.con.cursor()
        cur.execute(command)
        rows = cur.fetchall()
        ret = [(row[0], row[1], row[2]) for row in rows]
        cur.close()
        return ret
    
    def getCarTypesList(self):
        command = "select distinct(car_type_name) from car_types"
        rows = self.con.cursor().execute(command).fetchall()
        return [row[0] for row in rows]

    def getLicenseList(self):
        command = "select license_number from cars ORDER BY strftime('%Y%m%d%H%M%S', time) DESC"
        rows = self.con.cursor().execute(command).fetchall()
        return [row[0] for row in rows]
    
    def getRulesList(self):
        command = "select rule_name from rules"
        rows = self.con.cursor().execute(command).fetchall()
        return [row[0] for row in rows]

    # Insert
    def getMaxCarId(self):
        sql = '''select max(car_id) from cars'''
        carid = self.con.cursor().execute(sql).fetchall()[0][0]
        if carid is None:
            carid = 1
        return carid
    
    def insertIntoCars(self, car_type='', lic_img='', lic_num='', car_img=''):
        sql = '''INSERT INTO cars(car_type_id, license_image, license_number, car_image, time)
                      VALUES((SELECT car_type_id FROM car_types WHERE car_type_name = ? LIMIT 1),?,?,?, datetime()) '''

        car_img = car_img.split('/')[-1]
        lic_img = lic_img.split('/')[-1]
        cur = self.con.cursor()
        cur.execute(sql, (car_type, lic_img, lic_num, car_img))
        cur.close()
        self.con.commit()

    def insertIntoViolations(self,camera_group, camera, license_number, rule_name, time):
        sql = '''INSERT INTO violations(camera_id, car_id, rule_id, time)
                      VALUES(
                        (SELECT distinct(c.cam_id) from camera c 
                        INNER JOIN camera_group cg ON c.group_id = cg.group_id
                        WHERE cg.group_name LIKE ?
                        AND c.cam_name LIKE ?)
                        , (SELECT distinct(car_id) from cars
                            WHERE license_number LIKE ?)
                        , (SELECT distinct(rule_id) from rules
                            WHERE rule_name LIKE ?)
                        , ?) '''
        cur = self.con.cursor()
        cur.execute(sql, (camera_group, camera, license_number, rule_name, time))
        cur.close()
        self.con.commit()

    def insertIntoRules(self, rule, fine):
        sql = '''INSERT INTO rules(rule_name, rule_fine)
                      VALUES(?,?) '''
        cur = self.con.cursor()
        cur.execute(sql, (rule, fine))
        cur.close()
        self.con.commit()

    def insertIntoCamera(self, id, location, x, y, group, file):
        sql = '''INSERT INTO camera(id,location,coordinate_x, coordinate_y, feed, cam_group)
                      VALUES(?,?,?,?,?,?) '''
        file = file.split('/')[-1]
        cur = self.con.cursor()
        cur.execute(sql, (id, location, x, y, file, group))
        cur.close()
        self.con.commit()

    def search(self, cam=None, type=None, license=None, time=None):
        cur = self.con.cursor()
        command = "SELECT camera.location, cars.id, cars.color, cars.first_sighted, cars.license_image, " \
                  " cars.license_number, cars.car_image, cars.num_rules_broken, cars.owner," \
                  " rules.name, rules.fine, violations.time, rules.id" \
                  " FROM violations, rules, cars, camera" \
                  " where rules.id = violations.rule" \
                  " and violations.camera = camera.id" \
                  " and cars.id = violations.car"

        if cam is not None:
            command = command + " and violations.camera = '" + str(cam) + "'"
        if type is not None:
            command = command + " and cars.type_id = (select car_type_id from car_types where car_type_name='" + str(type) + "')"
        if time is not None:
            command = command + " and violations.time >= " + str(
                self.convertTimeToDB(time[0])) + " and violations.time <= " + str(self.convertTimeToDB(time[1]))

        cur.execute(command)
        rows = cur.fetchall()
        ret = []
        for row in rows:
            dict = {}
            dict[KEYS.LOCATION] = row[0]
            dict[KEYS.CARID] = row[1]
            dict[KEYS.CARTYPE] = row[2]

            carimage = QPixmap("car_images/" + row[4])
            dict[KEYS.CARIMAGE] = carimage

            dict[KEYS.LICENSENUMBER] = row[5]

            licenseimage = QPixmap("license_images/" + row[6])
            dict[KEYS.LICENSEIMAGE] = licenseimage

            dict[KEYS.CAROWNER] = row[8]
            dict[KEYS.RULENAME] = row[9]
            dict[KEYS.RULEFINE] = row[10]
            dict[KEYS.TIME] = row[11]
            dict[KEYS.RULEID] = row[12]
            ret.append(dict)
        cur.close()
        return ret

    def getViolationsFromCam(self, cam, cleared=False):
        cur = self.con.cursor()
        command = "SELECT camera.cam_location, cars.car_id, car_types.car_type_name, cars.car_image," \
                  " cars.license_number, cars.license_image," \
                  " rules.rule_name, rules.rule_fine, violations.time, rules.rule_id" \
                  " FROM violations" \
                  " INNER JOIN camera ON violations.camera_id = camera.cam_id" \
                  " INNER JOIN cars ON violations.car_id = cars.car_id" \
                  " INNER JOIN car_types ON cars.car_type_id = car_types.car_type_id" \
                  " INNER JOIN rules ON violations.rule_id = rules.rule_id" \
                  " WHERE 1=1"
        if cam is not None:
            command = command + " AND camera.cam_name = '" + str(cam) + "'"
        

        cur.execute(command)
        rows = cur.fetchall()
        ret = []
        for row in rows:
            dict = {}
            dict[KEYS.LOCATION] = row[0]
            dict[KEYS.CARID] = row[1]
            dict[KEYS.CARTYPE] = row[2]

            carImagePath = "/Users/acemitto/Desktop/STUDY/ACE/CAPSTONE/vietnamese-traffic-system/Source/Traffic-Rule-Violation-Detection-System-master/rule_breakers/" + row[3]
            carimage = QPixmap(carImagePath)
            dict[KEYS.CARIMAGE] = carimage

            dict[KEYS.LICENSENUMBER] = row[4]

            licenseImagePath = "/Users/acemitto/Desktop/STUDY/ACE/CAPSTONE/vietnamese-traffic-system/Source/Traffic-Rule-Violation-Detection-System-master/rule_breakers/license_images/" + row[5]
            licenseImage = QPixmap(licenseImagePath)
            dict[KEYS.LICENSEIMAGE] = licenseImage

            dict[KEYS.RULENAME] = row[6]
            dict[KEYS.RULEFINE] = row[7]
            dict[KEYS.TIME] = row[8]
            dict[KEYS.RULEID] = row[9]
            ret.append(dict)
        cur.close()
        return ret

    def deleteViolation(self, carid, ruleid, time):
        cur = self.con.cursor()
        command = "update violations set cleared = true " \
                  "where car = " + str(carid) + " and rule = " + str(ruleid) + " and time = " + str(time)
        rowcount = cur.execute(command).rowcount
        print("Deleted " + str(rowcount) + " rows")
        cur.close()
        self.con.commit()

    def getCamDetails(self, cam_id):
        command = "select count(*) from violations where camera_id = '" + str(cam_id) + "'"
        cur = self.con.cursor()
        count = cur.execute(command).fetchall()[0][0]
        cur.close()

        command = "select cam_location, cam_feed from camera where cam_name = '" + str(cam_id) + "'"
        cur = self.con.cursor()
        res = cur.execute(command).fetchall()
        cam_location = None
        cam_feed = None
        cam_location, cam_feed = res[0]
        cur.close()
        return count, cam_location, cam_feed

    def deleteAllCars(self):
        commad = "delete from cars"
        cur = self.con.cursor()
        cur.execute(commad)
        cur.close()
        self.con.commit()

    def deleteAllViolations(self):
        commad = "delete from violations"
        cur = self.con.cursor()
        cur.execute(commad)
        cur.close()
        self.con.commit()
    
    def clearCamLog(self):
        command = "update violations set cleared = true"
        cur = self.con.cursor()
        cur.execute(command)
        cur.close()
        self.con.commit()

    def convertTimeToDB(self, time):
        pass

    def convertTimeToGUI(self, time):
        pass

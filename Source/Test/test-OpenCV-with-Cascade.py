# Import OpenCV module  
import cv2  
# Import pyplot from matplotlib as plt  
from matplotlib import pyplot as pltd  
import pathlib
# Opening the image from files  
name_export="/Users/acemitto/Desktop/STUDY/ACE/CAPSTONE/vietnamese-traffic-system/Source/Traffic-Rule-Violation-Detection-System-master/test_images/OpenCV/"
PATH_TO_TEST_IMAGES_DIR = pathlib.Path('/Users/acemitto/Desktop/STUDY/ACE/CAPSTONE/vietnamese-traffic-system/Source/Traffic-Rule-Violation-Detection-System-master/test_images/')
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
print(TEST_IMAGE_PATHS)

for image_path in TEST_IMAGE_PATHS:
    imageName = str(image_path)
    imaging = cv2.imread(str(image_path))  
    # Altering properties of image with cv2  
    imaging_gray = cv2.cvtColor(imaging, cv2.COLOR_BGR2GRAY)  
    imaging_rgb = cv2.cvtColor(imaging, cv2.COLOR_BGR2RGB)  
    # Importing Haar cascade classifier xml data  
    xml_data = cv2.CascadeClassifier('/Users/acemitto/Desktop/STUDY/ACE/CAPSTONE/vietnamese-traffic-system/Source/vehicle-speed-check-master/myhaar.xml')  
    # Detecting object in the image with Haar cascade classifier   
    detecting = xml_data.detectMultiScale(imaging_gray,   
                                    minSize = (30, 30))  
    # Amount of object detected  
    amountDetecting = len(detecting)  
    # Using if condition to highlight the object detected  
    if amountDetecting != 0:  
        for (a, b, width, height) in detecting:  
            cv2.rectangle(imaging_rgb, (a, b), # Highlighting detected object with rectangle  
                        (a + height, b + width),   
                        (0, 275, 0), 3)  

    # export
    cv2.imwrite(name_export + str(imageName[len(imageName)-5]) + ".jpg",imaging_rgb)
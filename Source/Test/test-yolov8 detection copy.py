# Import OpenCV module  
import cv2  
# Import pyplot from matplotlib as plt  
from matplotlib import pyplot as pltd  
import pathlib
from ultralytics import YOLO

model = YOLO("yolov8x.pt")

# Opening the image from files  
name_export="/Users/acemitto/Desktop/STUDY/ACE/CAPSTONE/vietnamese-traffic-system/Source/Traffic-Rule-Violation-Detection-System-master/test_images/YOLO/"
PATH_TO_TEST_IMAGES_DIR = pathlib.Path('/Users/acemitto/Desktop/STUDY/ACE/CAPSTONE/vietnamese-traffic-system/Source/Traffic-Rule-Violation-Detection-System-master/test_images/')
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
print(TEST_IMAGE_PATHS)

for image_path in TEST_IMAGE_PATHS:
    imageName = str(image_path)
    imaging = cv2.imread(str(image_path))  
    # Altering properties of image with cv2  
    imaging_gray = cv2.cvtColor(imaging, cv2.COLOR_BGR2GRAY)  
    imaging_rgb = cv2.cvtColor(imaging, cv2.COLOR_BGR2RGB)  
    
    result = model(imaging, device="mps")[0]
    
    for box in result.boxes:
        class_id = result.names[box.cls[0].item()]
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        conf = round(box.conf[0].item(), 2)
        print("Object type:", class_id)
        print("Coordinates:", cords)
        print("Probability:", conf)
        print("---")
    
    # export
    cv2.imwrite(name_export + str(imageName[len(imageName)-5]) + ".jpg",imaging_rgb)
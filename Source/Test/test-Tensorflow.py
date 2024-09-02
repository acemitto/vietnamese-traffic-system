import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import zipfile
import time
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import imutils
from vehicle import vehicle
import PIL.Image as Image
from collections import defaultdict
from io import StringIO
from PIL import Image
import time
from multiprocessing.pool import ThreadPool
import threading
import time
import openalpr_api
from openalpr_api.rest import ApiException
import numpy as np
import cv2
import tkinter as tk
from PIL import Image
from PIL import ImageTk
import json
import re
import pathlib

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


from utils import label_map_util

from utils import visualization_utils as vis_util


name_export="/Users/acemitto/Desktop/STUDY/ACE/CAPSTONE/vietnamese-traffic-system/Source/Traffic-Rule-Violation-Detection-System-master/test_images/TF/"

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = pathlib.Path('/Users/acemitto/Desktop/STUDY/ACE/CAPSTONE/vietnamese-traffic-system/Source/Traffic-Rule-Violation-Detection-System-master/test_images/')
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
TEST_IMAGE_PATHS
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = '/Users/acemitto/Desktop/STUDY/ACE/STOCK/testData/Source/Traffic-Rule-Violation-Detection-System-master/Cars/output_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '/Users/acemitto/Desktop/STUDY/ACE/CAPSTONE/vietnamese-traffic-system/Source/Traffic-Rule-Violation-Detection-System-master/Cars/car_label_map.pbtxt'

NUM_CLASSES = 1

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def matchVehicles(currentFrameVehicles,im_width,im_height,image):
	if len(vehicles)==0:
		for box,color in currentFrameVehicles:
			(y1,x1,y2,x2)=box
			(x,y,w,h)=(x1*im_width,y1*im_height,x2*im_width-x1*im_width,y2*im_height-y1*im_height)
			X=int((x+x+w)/2)
			Y=int((y+y+h)/2)
			if Y>yl5:
				#cv2.circle(image,(X,Y),2,(0,255,0),4)
				#print('Y=',Y,'  y1=',yl1)
				vehicles.append(vehicle((x,y,w,h)))
	else:
		for i in range(len(vehicles)):
			vehicles[i].setCurrentFrameMatch(False)
			vehicles[i].predictNext()
		for box,color in currentFrameVehicles:
			(y1,x1,y2,x2)=box
			(x,y,w,h)=(x1*im_width,y1*im_height,x2*im_width-x1*im_width,y2*im_height-y1*im_height)
			#print((x1*im_width,y1*im_height,x2*im_width,y2*im_height),'\n',(x,y,w,h))
			index = 0
			ldistance = 999999999999999999999999.9
			X=int((x+x+w)/2)
			Y=int((y+y+h)/2)
			if Y>yl5:
				#print('Y=',Y,'  y1=',yl1)
				#cv2.circle(image,(X,Y),4,(0,0,255),8)
				for i in range(len(vehicles)):
					if vehicles[i].getTracking() == True:
						#print(vehicles[i].getNext(),i)
						distance = ((X-vehicles[i].getNext()[0])**2+(Y-vehicles[i].getNext()[1])**2)**0.5

						if distance<ldistance:
							ldistance = distance
							index = i


				diagonal=vehicles[index].diagonal

				if ldistance < diagonal:
					vehicles[index].updatePosition((x,y,w,h))
					vehicles[index].setCurrentFrameMatch(True)
				else:
					
					vehicles.append(vehicle((x,y,w,h)))

		for i in range(len(vehicles)):
			if vehicles[i].getCurrentFrameMatch() == False:
				vehicles[i].increaseFrameNotFound()
				
detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.compat.v1.GraphDef()
	with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')

vehicles=[]
for image_path in TEST_IMAGE_PATHS:
  imageName = str(image_path)
  image_np = cv2.imread(str(image_path))
  im_height, im_width, c = image_np.shape
  xl1=0
  xl2=im_width-1
  yl1=600
  yl2=yl1
  ml1=(yl2-yl1)/(xl2-xl1)
  intcptl1=yl1-ml1*xl1

  count=0
  xl3=0
  xl4=im_width-1
  yl3=450
  yl4=yl3
  ml2=(yl4-yl3)/(xl4-xl3)
  intcptl2=yl3-ml2*xl3

  xl5=0
  xl6=im_width-1
  yl5=0
  yl6=yl5
  ml3=(yl6-yl5)/(xl6-xl5)
  intcptl3=yl5-ml3*xl5
  ret=True
  start=time.time()
  c=0
  sesser=tf.Session(graph=detection_graph)
  image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
  # Each box represents a part of the image where a particular object was detected.
  detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
  # Each score represent how level of confidence for each of the objects.
  # Score is shown on the result image, together with the class label.
  detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
  detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
  num_detections = detection_graph.get_tensor_by_name('num_detections:0')




  image_np_expanded = np.expand_dims(image_np, axis=0)
  # Actual detection.
  (boxes, scores, classes, num) = sesser.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_np_expanded})


  # Visualization of the results of a detection.
  img=image_np
  imgF,coords=vis_util.visualize_boxes_and_labels_on_image_array(
  image_np,
  np.squeeze(boxes),
  np.squeeze(classes).astype(np.int32),
  np.squeeze(scores),
  category_index,
  use_normalized_coordinates=True,
  line_thickness=2)

  matchVehicles(coords,im_width,im_height,imgF)
  for v in vehicles:
    if v.getTracking()==True:

      for p in v.getPoints():
        cv2.circle(image_np,p,3,(200,150,75),6)

  # export

  cv2.imwrite(name_export + str(imageName[len(imageName)-5]) + ".jpg",image_np)
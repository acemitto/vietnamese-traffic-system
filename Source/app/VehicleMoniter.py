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

if tf.__version__ < '1.4.0':
	raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')
# This is needed to display the images.


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


from utils import label_map_util

from utils import visualization_utils as vis_util


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = '/Users/acemitto/Desktop/STUDY/ACE/STOCK/testData/Source/Traffic-Rule-Violation-Detection-System-master/Cars/output_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '/Users/acemitto/Desktop/STUDY/ACE/STOCK/testData/Source/Traffic-Rule-Violation-Detection-System-master/Cars/car_label_map.pbtxt'

NUM_CLASSES = 1

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
					#blue for last position
					#cv2.circle(image,tuple(vehicles[index].points[-1]),2,(255,0,0),4)
					#red for predicted point
					#cv2.circle(image,tuple(vehicles[index].getNext()),2,(0,0,255),2)
					#green for test point
					#cv2.circle(image,(X,Y),2,(0,255,0),4)

					#cv2.imshow('culprit',image)
					#time.sleep(5)
					#print(diagonal,'               ',ldistance)
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


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

cap=cv2.VideoCapture('/Users/acemitto/Desktop/STUDY/ACE/STOCK/testData/Source/Traffic-Rule-Violation-Detection-System-master/VietnamTraffic720p.mp4') # 0 stands for very first webcam attach
filename="testoutput.avi"
codec=cv2.VideoWriter_fourcc('m','p','4','v')#fourcc stands for four character code
framerate=24
resolution=(1280,720)

VideoFileOutput=cv2.VideoWriter(filename,codec,framerate, resolution)
vs = WebcamVideoStream(src='/Users/acemitto/Desktop/STUDY/ACE/STOCK/testData/Source/Traffic-Rule-Violation-Detection-System-master/VietnamTraffic720p.mp4').start()
ret,imgF=cap.read(0)
imgF=Image.fromarray(imgF)
im_width, im_height = imgF.size
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
yl5=300
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

window = tk.Tk()  #Makes main window
window.wm_title("T.M.S")
window.columnconfigure(0, {'minsize': 1020})
window.columnconfigure(1, {'minsize': 335})


frame=tk.Frame(window)
frame.grid(row=0,column=0,rowspan=5,sticky='N',pady=10)

frame2=tk.Frame(window)
frame2.grid(row=0,column=1)

frame3=tk.Frame(window)
frame3.grid(row=1,column=1)

frame4=tk.Frame(window)
frame4.grid(row=2,column=1)

frame5=tk.Frame(window)
frame5.grid(row=3,column=1)

frame2.rowconfigure(1, {'minsize': 250})
frame3.rowconfigure(1, {'minsize': 80})
frame4.rowconfigure(1, {'minsize': 150})
frame5.rowconfigure(1, {'minsize': 80})

vehicles=[]
def main(sess=sesser):
	'''global masterframe
	global started'''
	if True:
		fTime=time.time()
		_,image_np=cap.read(0)
		print(image_np)
		#image_np = imutils.resize(image_np, width=400)

		# Definite input and output Tensors for detection_graph


		# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
		image_np_expanded = np.expand_dims(image_np, axis=0)
		# Actual detection.
		(boxes, scores, classes, num) = sess.run(
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
			line_thickness=1)

		matchVehicles(coords,im_width,im_height,imgF)
		for v in vehicles:
			if v.getTracking()==True:

				for p in v.getPoints():
					cv2.circle(image_np,p,3,(200,150,75),6)

			#print(ymin*im_height,xmin*im_width,ymax*im_height,xmax*im_width)
			#cv2.rectangle(image_np,(int(xmin*im_width),int(ymin*im_height)),(int(xmax*im_width),int(ymax*im_height)),(255,0,0),2)
		VideoFileOutput.write(image_np)
		#print('yola')
		frame=cv2.resize(image_np,(1020,647))
		cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
		img = Image.fromarray(cv2image)
		imgtk = ImageTk.PhotoImage(image=img)
		display1.imgtk = imgtk #Shows frame for display 1
		display1.configure(image=imgtk)
	window.after(1, main)


lbl1 = tk.Label(frame,text='Vehicle Detection And Tracking',font = "verdana 12 bold")
lbl1.pack(side='top')

lbl2 = tk.Label(frame2,text='Vehicle Breaking Traffic Rule',font = "verdana 10 bold")
lbl2.grid(row=0,column=0,sticky ='S',pady=10)

lbl3 = tk.Label(frame3,text='Veicle Speed',font = "verdana 10 bold")
lbl3.grid(row=0,column=0,sticky ='S',pady=10)


lbl4 = tk.Label(frame4,text='Detected License Plate',font = "verdana 10 bold")
lbl4.grid(row=0,column=0)

lbl5 = tk.Label(frame5,text='Extracted License Plate Number',font = "verdana 10 bold")
lbl5.grid(row=0,column=0)

display1 = tk.Label(frame)
display1.pack(side='bottom')  #Display 1

display2 = tk.Label(frame2)
display2.grid(row=1,column=0) #Display 2


display3 = tk.Label(frame3,text="",font = "verdana 14 bold",fg='red')
display3.grid(row=1,column=0)

display4 = tk.Label(frame4)
display4.grid(row=1,column=0)

display5 = tk.Label(frame5,text="",font = "verdana 24 bold",fg='green')
display5.grid(row=1,column=0)
masterframe=None
started= False
def stream():
    global masterframe
    global started
    global c
    global tim
    cap=cv2.VideoCapture('/Users/acemitto/Desktop/STUDY/ACE/STOCK/testData/Source/Traffic-Rule-Violation-Detection-System-master/VietnamTraffic720p.mp4')
    while True:
        started,masterframe=cap.read()
        time.sleep(0.034)


'''thread = threading.Thread(target=stream)
thread.daemon = True
thread.start()'''
with detection_graph.as_default():
	with tf.Session(graph=detection_graph) as sess:
		sesser=sess
		main(sess) #Display
window.mainloop()  #Starts GUI

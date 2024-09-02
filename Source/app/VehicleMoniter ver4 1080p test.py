import torch
from numpy import random
import math
import cv2
import time
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque
from shapely.geometry import Polygon
from shapely.geometry import Point
import numpy as np
import yolov7
import supervision as sv
import time
from datetime import datetime
import csv
import PIL.Image as Image
from PIL import ImageTk
from PIL import Image
from local_utils import detect_lp
import tkinter as tk
from os.path import splitext
from keras.models import model_from_json
from paddleocr import PaddleOCR, draw_ocr # main OCR dependencies

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
speed_line_queue = {}
cross_line_queue = {}
data_deque = {}
deepsort = None

with open('/Users/acemitto/Desktop/traffic_measurement_tw.csv', 'w') as f:
    writer = csv.writer(f)
    csv_line = \
        'no|Day|Hour|Time|Location|Type|Speed|Status'
    writer.writerows([csv_line.split('|')])

def init_tracker():
	global deepsort
	cfg_deep = get_config()
	cfg_deep.merge_from_file("/Users/acemitto/Desktop/STUDY/ACE/STOCK/testData/Source/Traffic-Rule-Violation-Detection-System-master/deep_sort_pytorch/configs/deep_sort.yaml")

	deepsort= DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
							max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
							nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
							max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
							use_cuda=True)
##########################################################################################

def load_model(path):
	try:
		path = splitext(path)[0]
		with open('%s.json' % path, 'r') as json_file:
			model_json = json_file.read()
		model = model_from_json(model_json, custom_objects={})
		model.load_weights('%s.h5' % path)
		print("Loading model successfully...")
		return model
	except Exception as e:
		print(e)

def preprocess_image(image_path):
	img = cv2.imread(image_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = img / 255
	return img

wpod_net_path = "/Users/acemitto/Desktop/STUDY/ACE/STOCK/testData/Source/Traffic-Rule-Violation-Detection-System-master/wpod-net.json"
wpod_net = load_model(wpod_net_path)

# Calculation the speed of vehicles
def estimate_speed(timeCrossedTheGreenLight, timeCrossedTheBlueLight):
	d_time = (timeCrossedTheGreenLight - timeCrossedTheBlueLight) /1000
	print(timeCrossedTheGreenLight - timeCrossedTheBlueLight)
	d_meters = 4
	speed = d_meters / d_time * 3.6
	return int(speed)

# convert the coordinates of the bounding box to the center coordinates of the bounding box
def xyxy_to_xywh(*xyxy):
	"""" Calculates the relative bounding box from absolute pixel values. """
	bbox_left = min([xyxy[0].item(), xyxy[2].item()])
	bbox_top = min([xyxy[1].item(), xyxy[3].item()])
	bbox_w = abs(xyxy[0].item() - xyxy[2].item())
	bbox_h = abs(xyxy[1].item() - xyxy[3].item())
	x_c = (bbox_left + bbox_w / 2)
	y_c = (bbox_top + bbox_h / 2)
	w = bbox_w
	h = bbox_h
	return x_c, y_c, w, h

def overlap(rec1, rec2):
	if (rec2[2] > rec1[0] and rec2[2] < rec1[2]) or \
		(rec2[0] > rec1[0] and rec2[0] < rec1[2]):
		x_match = True
	else:
		x_match = False
	if (rec2[3] > rec1[1] and rec2[3] < rec1[3]) or \
		(rec2[1] > rec1[1] and rec2[1] < rec1[3]):
		y_match = True
	else:
		y_match = False
	if x_match and y_match:
		return True
	else:
		return False

def overlap2(rect1,rect2):
    p1 = Polygon([(rect1[0],rect1[1]), (rect1[1],rect1[1]),(rect1[2],rect1[3]),(rect1[2],rect1[1])])
    p2 = Polygon([(rect2[0],rect2[1]), (rect2[1],rect2[1]),(rect2[2],rect2[3]),(rect2[2],rect2[1])])
    return(p1.intersects(p2))

def compute_color_for_labels(label):
	"""
	Simple function that adds fixed color depending on the class
	"""
	if label == 0: #person
		color = (85,45,255)
	elif label == 2: # Car
		color = (222,82,175)
	elif label == 3:  # Motobike
		color = (0, 204, 255)
	elif label == 5:  # Bus
		color = (0, 149, 255)
	else:
		color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
	return tuple(color)

def detect_red_and_yellow(rlimg):
    # convert BGR image to HSV
    desired_dim = (90, 30)  # width, height
    rlimg = cv2.cvtColor(rlimg, cv2.COLOR_BGR2RGB)
    rlimg = cv2.resize(np.array(rlimg), desired_dim, interpolation=cv2.INTER_LINEAR)
    hsv_img = cv2.cvtColor(rlimg, cv2.COLOR_RGB2HSV)

    # min and max HSV values
    red_min = np.array([0,5,150])
    red_max = np.array([8,255,255])
    red_min2 = np.array([175,5,150])
    red_max2 = np.array([180,255,255])

    yellow_min = np.array([20,5,150])
    yellow_max = np.array([30,255,255])

    green_min = np.array([35,5,150])
    green_max = np.array([90,255,255])

    # apply red, yellow, green thresh to image
    # 利用cv2.inRange函数设阈值，去除背景部分
    red_thresh = cv2.inRange(hsv_img,red_min,red_max)+cv2.inRange(hsv_img,red_min2,red_max2)
    yellow_thresh = cv2.inRange(hsv_img,yellow_min,yellow_max)
    green_thresh = cv2.inRange(hsv_img,green_min,green_max)

    # apply blur to fix noise in thresh
    # 进行中值滤波
    red_blur = cv2.medianBlur(red_thresh,5)
    yellow_blur = cv2.medianBlur(yellow_thresh,5)
    green_blur = cv2.medianBlur(green_thresh,5)

    # checks which colour thresh has the most white pixels
    red = cv2.countNonZero(red_blur)
    yellow = cv2.countNonZero(yellow_blur)
    green = cv2.countNonZero(green_blur)
    # the state of the light is the one with the greatest number of white pixels
    lightColor = max(red,yellow,green)

    # pixel count must be greater than 60 to be a valid colour state (solid light or arrow)
    # since the ROI is a rectangle that includes a small area around the circle
    # which can be detected as yellow
    if lightColor > 60:
        if lightColor == red:
            return 1
        elif lightColor == yellow:
            return 2
        elif lightColor == green:
            return 3
    else:
        return 0

stt_all = 0
def getDataCrossedTheLight(id, data_deque, object_speed, obj_name):
	global stt_all
	if data_deque[1][1] >= yl2 and data_deque[0][1] < yl2:
		stt_all = stt_all + 1
		status = str(stt_all)
		time_obj = str(time.asctime(time.localtime(time.time())))
		location = 'N/A'
		vehicle_type = str(obj_name)
		vehicle_speed = str(object_speed)
		data_signal = 'OK'
        
		if csv_line != 'not_available':
			# no|Day|Hour|Time|Location|Type|Speed|Status
			with open('/Users/acemitto/Desktop/traffic_measurement_tw.csv', 'a') as f:
				writer = csv.writer(f)
				csv_line_queue = status+'|2023-12-10|0|'+time_obj+'|'+location+'|'+vehicle_type+'|'+vehicle_speed+'|'+data_signal
				writer.writerows([csv_line_queue.split('|')])

	if data_deque[1][1] < yl2 and data_deque[0][1] >= yl2:
		stt_all = stt_all + 1
		status = str(stt_all)
		time_obj = str(time.asctime(time.localtime(time.time())))
		location = 'N/A'
		vehicle_type = str(obj_name)
		vehicle_speed = str(object_speed)
		data_signal = 'OK'
        
		if csv_line != 'not_available':
			# no|Day|Hour|Time|Location|Type|Speed|Status
			with open('/Users/acemitto/Desktop/traffic_measurement_tw.csv', 'a') as f:
				writer = csv.writer(f)
				csv_line_queue = status+'|2023-12-10|0|'+time_obj+'|'+location+'|'+vehicle_type+'|'+vehicle_speed+'|'+data_signal
				writer.writerows([csv_line_queue.split('|')])


def checkCrossedTheBlueLine(data_deque, direction_obj):
	if direction_obj == 'up':
		if data_deque[1][1] >= yj2 and data_deque[0][1] < yj2:
			return True
		else :
			return False
	elif  direction_obj == 'down':
		if data_deque[1][1] < yj2 and data_deque[0][1] >= yj2:
			return True
		else :
			return False
	
def checkCrossedTheGreenLine(data_deque, direction_obj):
	if direction_obj == 'up':
		if data_deque[1][1] >= yz2 and data_deque[0][1] < yz2:
			return True
		else :
			return False
	elif  direction_obj == 'down':
		if data_deque[1][1] < yz2 and data_deque[0][1] >= yz2:
			return True
		else :
			return False
	
def checkWrongLane(data_deque, direction):
	polygon_left = Polygon(polygons[0])
	polygon_right = Polygon(polygons[1])
	point = Point(data_deque[0])

	if direction == 'down':
		if polygon_right.contains(point):
			return "Wrong Lane"
	elif direction == 'up':
		if polygon_left.contains(point):
			return "Wrong Lane"
	else:
		return False

def getDirection(data_deque):
	up_down = data_deque[4][1] - data_deque[0][1]
	left_right = data_deque[4][0] - data_deque[0][0]
	up = 0
	down = 0
	left = 0
	right = 0

	if up_down > 0:
		up = up_down
	else:
		down = abs(up_down)
	
	if left_right > 0:
		left = left_right
	else:
		left = abs(left_right)

	direction = max(up,down,left,right)
	if direction > 5:
		if direction == up:
			return 'up'
		elif direction == down:
			return 'down'
		elif direction == left:
			return 'left'
		elif direction == right:
			return 'right'
	else:
		return 'stay'

# draw the bounding box and the speed of the object
def UI_box(x, img, kmh, obj_name, data_deque, red_light, wrong_lane, color=None, label=None, id=id, line_thickness=None):
	tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
	color = color or [random.randint(0, 255) for _ in range(3)]
	c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
	w, h = int(x[2] - x[0]), int(x[3] - x[1])
	cx, cy = int(x[0] + w / 2), int(x[1] + h / 2)
	bimg=img[int(x[1]):int(x[3]), int(x[0]):int(x[2])]
	cv2.circle(img, (cx, cy), 2, (0,0,255), cv2.FILLED)
	img = cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

	if label:
		tf = max(tl - 1, 1)  # font thickness
		t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
		
		# img = cv2.rectangle(img, (c1[0], c1[1] - t_size[1] - 3), (c1[0] + t_size[0], c1[1] - 2), color, -1, cv2.LINE_AA)  # filled
		cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
		
	if wrong_lane:
		t_size = cv2.getTextSize(wrong_lane, 0, fontScale=0.5, thickness=1)[0]
		img = cv2.rectangle(img, (c1[0], c1[1] - t_size[1] - 3), (c1[0] + t_size[0], c1[1]), [0,0,255], -1, cv2.LINE_AA)  # filled
		cv2.putText(img, wrong_lane, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 0], thickness=1, lineType=cv2.LINE_AA)

		
		

# draw the bounding box and the speed of the object
def draw_boxes(img, bbox, names, object_id, cur_time, identities=None, offset=(0, 0)):
	height, width, _ = img.shape
	for key in list(data_deque):
		if key not in identities:  
			data_deque.pop(key)
	redlightimg = img[int(64):int(90), int(1069):int(1160)]
	red_light = detect_red_and_yellow(redlightimg)
	if red_light == 1:
		cv2.putText(img, 'Red', (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
	elif red_light == 2:
		cv2.putText(img, 'Yellow', (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
	elif red_light == 3:
		cv2.putText(img, 'Green', (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

	for i, box in enumerate(bbox):
		x1, y1, x2, y2 = [int(i) for i in box]
		x1 += offset[0]
		x2 += offset[0]
		y1 += offset[1]
		y2 += offset[1]
		speed_check = 0
		# bounding box center
		center = int(x1+((x2-x1) / 2)), int(y1+(y2 - y1) / 2)
		

		# get ID of object
		id = int(identities[i]) if identities is not None else 0
			
		# create new buffer for new object
		if id not in data_deque:  
			data_deque[id] = deque(maxlen= 64)
			speed_line_queue[id] = []
			cross_line_queue[id] = []
	
		color = compute_color_for_labels(object_id[i])
		obj_name = names[object_id[i]]

		label = '{}{:d}'.format("", id) + ": "+ '%s' % (obj_name)
		# add center to buffer
		data_deque[id].appendleft(center)
		# if data deque has more than two value, calculate speed

		if len(data_deque[id]) >= 2:
			wrong_lane = ''
			direction_obj = ''
			if len(data_deque[id]) >= 5:
				direction_obj = getDirection(data_deque[id])
				wrong_lane = checkWrongLane(data_deque[id], direction_obj)
				label = label + " " + str(direction_obj)

			object_speed = 0
			if checkCrossedTheBlueLine(data_deque[id], direction_obj):
				if direction_obj == 'up':
					cross_line_queue[id].append(cur_time)
					print("crossBLueLight: "+ str(id) + " " + str(cross_line_queue[id]))
				elif direction_obj == 'down':
					print("crossBLueLight: "+ str(id))
					if int(sum(cross_line_queue[id])) != 0:
						print("Da co data: " + str(id) + " " + str(cross_line_queue[id]))
						object_speed = estimate_speed(cur_time, sum(cross_line_queue[id]))
						print(object_speed)
						speed_line_queue[id].append(object_speed)
						# getDataCrossedTheLight(id, object_speed, obj_name)

			if checkCrossedTheGreenLine(data_deque[id], direction_obj):
				if direction_obj == 'up':
					print("crossGreenLight: "+ str(id))
					if int(sum(cross_line_queue[id])) != 0:
						print("Da co data: " + str(id) + " " + str(cross_line_queue[id]))
						object_speed = estimate_speed(cur_time, sum(cross_line_queue[id]))
						print(object_speed)
						speed_line_queue[id].append(object_speed)
						# getDataCrossedTheLight(id, object_speed, obj_name)
				elif direction_obj == 'down':
					cross_line_queue[id].append(cur_time)
					print("crossBLueLight: "+ str(id) + " " + str(cross_line_queue[id]))
				
			getDataCrossedTheLight(id, data_deque[id], object_speed, obj_name)
			if  int(sum(speed_line_queue[id])) != 0:
				label = label + " " + str(int(sum(speed_line_queue[id]))) + "km/h"
			UI_box(box, img, speed_check, obj_name, data_deque[id], red_light, wrong_lane, label=label, color=color, id=id ,line_thickness=1)
		
		# draw trail
		for i in range(1, len(data_deque[id])):
			# check if on buffer value is none
			if data_deque[id][i - 1] is None or data_deque[id][i] is None:
				continue
			# generate dynamic thickness of trails
			thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
			# draw trails
			cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color, thickness)

	return img

init_tracker()

# load yolov7 model
# pip install yolov7detect
model = yolov7.load("yolov7-e6e.pt",trace=False)#),device="cuda:0")
print("model loaded")
print("Names: ", model.names)

colors = sv.ColorPalette.default()
# initiate polygon zone
polygons = [
	np.array([
		[915, 570],[1094, 570],[1100, 1080],[540, 1080]
	]),
	np.array([
		[1104, 570],[1292, 570],[1707, 1080],[1126, 1080]
	])
]

# polygons = [ np.array([
#	 [0, 350],[1920, 350],[1920, 1080],[0, 1080]
# ])]

cap=cv2.VideoCapture('/Users/acemitto/Desktop/no audio Huwei/video 4 1080p.mp4') # 0 stands for very first webcam attach
w,h = int(cap.get(3)), int(cap.get(4))
ret,imgF=cap.read(0)
imgF=Image.fromarray(imgF)
fps = cap.get(cv2.CAP_PROP_FPS)
print("fps: ", cap.get(cv2.CAP_PROP_FPS))

# filename="/Users/acemitto/Desktop/testttttttt.mp4"
# codec=cv2.VideoWriter_fourcc('m','p','4','v')#fourcc stands for four character code
# resolution=(1920,1080)
# VideoFileOutput=cv2.VideoWriter(filename,codec,fps, resolution)

im_width, im_height = imgF.size
xl1=0
xl2=im_width-1
yl1=722
yl2=yl1

xj1=0
xj2=im_width-1
yj1=778
yj2=yj1

xz1=0
xz2=im_width-1
yz1=809
yz2=yz1

zones = [
	sv.PolygonZone(
		polygon=polygon, 
		frame_resolution_wh=(w,h)
	)
	for polygon
	in polygons
]

zone_annotators = [
	sv.PolygonZoneAnnotator(
		zone=zone, 
		color=colors.by_idx(index), 
		thickness=2
	)
	for index, zone in enumerate(zones)
]

box_annotators = [
	sv.BoxAnnotator(
		color=colors.by_idx(index), 
		thickness=1, 
		text_thickness=1, 
		text_scale=0.5
		)
	for index
	in range(len(polygons))
]

window = tk.Tk()  #Makes main window
window.wm_title("Tracking Giao Thong")
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
def main(sess=''):
	if True:
		fTime=time.time()
		ret, frame = cap.read(0)
		# detect
		results = model(frame)
		detections = sv.Detections.from_yolov5(results)
		detections = (detections[(detections.class_id != 6)])
		# 2,3,5 = person, car, bus just detect these objects

		xywh_bboxs = []
		confs = []
		oids = []
		outputs = []
		for xyxy, conf, cls in zip(detections.xyxy, detections.confidence, detections.class_id):
			x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
			xywh_obj = [x_c, y_c, bbox_w, bbox_h]
			xywh_bboxs.append(xywh_obj)
			confs.append([conf.item()])
			oids.append(int(cls))
		xywhs = torch.Tensor(xywh_bboxs)
		confss = torch.Tensor(confs)
		if xywhs.nelement() > 0:
			outputs = deepsort.update(xywhs, confss, oids, frame)
		if len(outputs) > 0:
			bbox_xyxy = outputs[:, :4]
			identities = outputs[:, -2]
			object_id = outputs[:, -1]
			frame = draw_boxes(frame, bbox_xyxy, model.names, object_id, cap.get(cv2.CAP_PROP_POS_MSEC), identities)
		
		for i, (zone, zone_annotator, box_annotator) in enumerate(zip(zones, zone_annotators, box_annotators)):
			# annotate
			frame = zone_annotator.annotate(scene=frame)
		
		#print('yola')
		cv2.line(frame, (int(xl1),int(yl1)), (int(xl2),int(yl2)), (0,255,0),1)
		cv2.line(frame, (int(xj1),int(yj1)), (int(xj2),int(yj2)), (255,0,0),1)
		cv2.line(frame, (int(xz1),int(yz1)), (int(xz2),int(yz2)), (0,0,255),1)
		# VideoFileOutput.write(frame)
		frame_rs=cv2.resize(frame,(1020,647))
		cv2image = cv2.cvtColor(frame_rs, cv2.COLOR_BGR2RGBA)
		img = Image.fromarray(cv2image)
		imgtk = ImageTk.PhotoImage(image=img)
		display1.imgtk = imgtk #Shows frame for display 1
		display1.configure(image=imgtk)
	window.after(1, main)


lbl1 = tk.Label(frame,text='Vehicle Detection And Tracking',font = "verdana 12 bold")
lbl1.pack(side='top')

lbl2 = tk.Label(frame2,text='Vehicle Breaking Traffic Rule',font = "verdana 10 bold")
lbl2.grid(row=0,column=0,sticky ='S',pady=10)

lbl3 = tk.Label(frame3,text='Vehicle Speed',font = "verdana 10 bold")
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

main('') #Display
window.mainloop()  #Starts GUI


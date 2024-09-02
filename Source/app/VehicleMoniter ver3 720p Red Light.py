import torch
from numpy import random
import math
import cv2
import time
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque
import numpy as np
import yolov7
import supervision as sv
import time

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
red_light = False

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

def get_plate(image_path, vehicle_type):
	if vehicle_type == 'motorcycle':
		Dmax = 300
		Dmin = 600
	else:
		Dmax = 300
		Dmin = 800

	vehicle = preprocess_image(image_path)
	ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
	side = int(ratio * Dmin)
	bound_dim = min(side, Dmax)
	_ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, vehicle_type, lp_threshold=0.1)
	return vehicle, LpImg, cor

# sử dụng WPOD-NET để crop biển số và detect ký tự
def getLicensePlateNumber(url, obj_name):
	path = url
	vehicle, LpImg, cor = get_plate(path, obj_name)
	display4.configure(image='')
	display5.configure(text="")

	if (len(LpImg)): # check if there is at least one license image
		# Scales, calculates absolute values, and converts the result to 8-bit.
		plate_image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
		# convert to grayscale and blur the image
		gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
		blur = cv2.GaussianBlur(gray,(7,7),0)

		img=blur
		frame3=img
		img3 = Image.fromarray(frame3)
		w,h=img3.size
		asprto=w/h
		frame3=cv2.resize(frame3,(150,int(150/asprto)))
		cv2image3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGBA)
		img3 = Image.fromarray(cv2image3)
		imgtk3 = ImageTk.PhotoImage(image=img3)
		display4.imgtk = imgtk3 #Shows frame for display 1
		display4.configure(image=imgtk3)

	licensePlate = ""
	if cor != '' :
		ocr_model = PaddleOCR(use_angle_cls=False, lang='en', show_log = False, use_gpu=True)
		result2 = ocr_model.ocr(blur)

		for i in range(len(result2[0])):
			if result2[0][i][1][0] != "" :
				licensePlate += result2[0][i][1][0] + ' '
				print(result2[0][i][1][0])

		display5.configure(text=licensePlate)
	else :
		display5.configure(text="cant detected!")
		licensePlate = "cant detected!"
	return licensePlate
				
# Calculation the speed of vehicles
def estimate_speed(timeCrossedTheGreenLight, timeCrossedTheBlueLight):
	d_time = (timeCrossedTheGreenLight - timeCrossedTheBlueLight) /1000
	print(timeCrossedTheGreenLight - timeCrossedTheBlueLight)
	d_meters = 5
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

def detect_red_and_yellow(img, Threshold=0.01):
	"""
	detect red and yellow
	:param img:
	:param Threshold:
	:return:
	"""

	desired_dim = (30, 90)  # width, height
	img = cv2.resize(np.array(img), desired_dim, interpolation=cv2.INTER_LINEAR)
	img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

	# lower mask (0-10)
	lower_red = np.array([0, 70, 50])
	upper_red = np.array([10, 255, 255])
	mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

	# upper mask (170-180)
	lower_red1 = np.array([170, 70, 50])
	upper_red1 = np.array([180, 255, 255])
	mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)

	# defining the Range of yellow color
	lower_yellow = np.array([21, 39, 64])
	upper_yellow = np.array([40, 255, 255])
	mask2 = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

	# red pixels' mask
	mask = mask0 + mask1 + mask2

	# Compare the percentage of red values
	rate = np.count_nonzero(mask) / (desired_dim[0] * desired_dim[1])

	if rate > Threshold:
		return True
	else:
		return False

def check_cross_redlight(light, bimg, obj_name, data_deque):
	if data_deque[1][1] >= yj2 and data_deque[0][1] < yj2:
		print(light)
		if light:
			name='/Users/acemitto/Desktop/STUDY/ACE/CAPSTONE/vietnamese-traffic-system/Source/Traffic-Rule-Violation-Detection-System-master/Rule Breakers/Vi Pham '+str(time.time())+'.jpg'
			h, w = bimg.shape[:2]
			frame2=bimg
			img2 = Image.fromarray(frame2)
			w,h=img2.size
			asprto=w/h
			frame2=cv2.resize(frame2,(250,int(250/asprto)))
			cv2image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGBA)
			img2 = Image.fromarray(cv2image2)
			imgtk2 = ImageTk.PhotoImage(image=img2)
			display2.imgtk = imgtk2 #Shows frame for display 1
			display2.configure(image=imgtk2)
			display3.configure(text='Crossed Red Light')

			cv2.imwrite(name,bimg)
			license_plate_queue = getLicensePlateNumber(name, obj_name)

def checkCrossedTheBlueLine(data_deque):
	if data_deque[1][1] >= yj2 and data_deque[0][1] < yj2:
		return True
	else :
		return False
	
def checkCrossedTheGreenLine(data_deque):
	if data_deque[1][1] >= yl2 and data_deque[0][1] < yl2:
		return True
	else :
		return False
									   
# draw the bounding box and the speed of the object
def UI_box(x, img, kmh, obj_name, data_deque, color=None, label=None, id=id, line_thickness=None):
	global red_light

	tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
	color = color or [random.randint(0, 255) for _ in range(3)]
	c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
	w, h = int(x[2] - x[0]), int(x[3] - x[1])
	cx, cy = int(x[0] + w / 2), int(x[1] + h / 2)
	bimg=img[int(x[1]):int(x[3]), int(x[0]):int(x[2])]

	if label:
		if label == "4: traffic light":
			red_light = detect_red_and_yellow(bimg)
			print(red_light)
			
		check_cross_redlight(red_light, bimg, obj_name, data_deque)
	
	cv2.circle(img, (cx, cy), 2, (0,0,255), cv2.FILLED)
	img = cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

	if label:
		tf = max(tl - 1, 1)  # font thickness
		t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
		
		# img = cv2.rectangle(img, (c1[0], c1[1] - t_size[1] - 3), (c1[0] + t_size[0], c1[1] - 2), color, -1, cv2.LINE_AA)  # filled
		cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
		

# draw the bounding box and the speed of the object
def draw_boxes(img, bbox, names, object_id, cur_time, identities=None, offset=(0, 0)):
	height, width, _ = img.shape
	for key in list(data_deque):
		if key not in identities:  
			data_deque.pop(key)
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
			if checkCrossedTheBlueLine(data_deque[id]):
				cross_line_queue[id].append(cur_time)
				print("crossBLueLight: "+ str(id) + " " + str(cross_line_queue[id]))
			if checkCrossedTheGreenLine(data_deque[id]):
				print("crossgreenLight: "+ str(id))
				if int(sum(cross_line_queue[id])) != 0:
					print("Da co data: " + str(id) + " " + str(cross_line_queue[id]))
					object_speed = estimate_speed(cur_time, sum(cross_line_queue[id]))
					print(object_speed)
					speed_line_queue[id].append(object_speed)
					# getDataCrossedTheLight(id, object_speed, obj_name)

			if  int(sum(speed_line_queue[id])) != 0:
				label = label + " " + str(int(sum(speed_line_queue[id]))) + "km/h"
			UI_box(box, img, speed_check, obj_name, data_deque[id], label=label, color=color, id=id ,line_thickness=1)
		
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
		[0, 503],[451, 511],[359, 720],[0, 720]
	]),
	np.array([
		[470, 511],[1120, 520],[1231, 720],[387, 720]
	])
]
# polygons = [ np.array([
#	 [0, 350],[1920, 350],[1920, 1080],[0, 1080]
# ])]
polygons2 = [ np.array([
	[520, 350],[1250, 350],[1920, 1010],[100, 1010]
])]
cap=cv2.VideoCapture('/Users/acemitto/Desktop/Camera2.mp4') # 0 stands for very first webcam attach
w,h = int(cap.get(3)), int(cap.get(4))
ret,imgF=cap.read(0)
imgF=Image.fromarray(imgF)
fps = cap.get(cv2.CAP_PROP_FPS)
print("fps: ", cap.get(cv2.CAP_PROP_FPS))

filename="/Users/acemitto/Desktop/Camera2-output.mp4"
codec=cv2.VideoWriter_fourcc('m','p','4','v')#fourcc stands for four character code
resolution=(1280,720)
VideoFileOutput=cv2.VideoWriter(filename,codec,fps, resolution)

im_width, im_height = imgF.size
xl1=0
xl2=im_width-1
yl1=395
yl2=yl1

xj1=0
xj2=im_width-1
yj1=471
yj2=yj1

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
	for index, zone
	in enumerate(zones)
]

box_annotators = [
	sv.BoxAnnotator(
		color=colors.by_idx(index), 
		thickness=1, 
		text_thickness=1, 
		text_scale=1
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
		detections = (detections[(detections.class_id != 0)])
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
			mask, idd = zone.trigger(detections=detections)
			#print(mask, idd)
			detections_filtered = detections[mask]
			detections_check = detections[mask]
			detections_check.class_id = []
			detections_check.xyxy = []

			for x in range(len(detections_filtered.class_id)) :
				if i == 1:
					if detections_filtered.class_id[x] != 3 :
						detections_check.class_id.append(detections_filtered.class_id[x])
						detections_check.xyxy.append(detections_filtered.xyxy[x])

			frame = box_annotator.annotate(scene=frame, detections=detections_check, skip_label=False)
			frame = zone_annotator.annotate(scene=frame)
			
		#print('yola')
		cv2.line(frame, (int(xl1),int(yl1)), (int(xl2),int(yl2)), (0,255,0),1)
		cv2.line(frame, (int(xj1),int(yj1)), (int(xj2),int(yj2)), (255,0,0),1)
		VideoFileOutput.write(frame)
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


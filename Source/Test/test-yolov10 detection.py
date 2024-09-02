from ultralytics import YOLO
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

def init_tracker():
	global deepsort
	cfg_deep = get_config()
	cfg_deep.merge_from_file("/Users/acemitto/Desktop/STUDY/ACE/CAPSTONE/vietnamese-traffic-system/Source/Traffic-Rule-Violation-Detection-System-master/deep_sort_pytorch/configs/deep_sort.yaml")

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

def getDirection(data_deque):
	up_down = data_deque[5][1] - data_deque[0][1]
	left_right = data_deque[5][0] - data_deque[0][0]
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
	if direction > 4.5:
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
def UI_box(x, img, obj_name, data_deque, color=None, label=None, id=id, line_thickness=None):
	tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
	color = color or [random.randint(0, 255) for _ in range(3)]
	c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
	w, h = int(x[2] - x[0]), int(x[3] - x[1])
	cx, cy = int(x[0] + w / 2), int(x[1] + h / 2)
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
			if len(data_deque[id]) >= 6:
				direction_obj = getDirection(data_deque[id])
				label = label + " " + str(direction_obj)

			if  int(sum(speed_line_queue[id])) != 0:
				label = label + " " + str(int(sum(speed_line_queue[id]))) + "km/h"
			UI_box(box, img, obj_name, data_deque[id], label=label, color=color, id=id ,line_thickness=1)
		
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

# load yolov10 model
# pip install yolov7detect
# Load a model
# model = YOLO("yolov10x.yaml")  # build a new model from scratch
model = YOLO("yolov10m.pt")  # load a pretrained model (recommended for training)

print("model loaded")
print("Names: ", model.names)

colors = sv.ColorPalette.default()
# initiate polygon zone
polygons = [
	np.array([
		[853, 24],[1036, 24],[846, 1080],[268, 1080]
	]),
	np.array([
		[1042, 24],[1228, 24],[1446, 1080],[865, 1080]
	]),
	np.array([
		[779, 139],[1022, 139],[1005, 241],[722, 241]
	])
]

cap=cv2.VideoCapture('/Users/acemitto/Desktop/STUDY/ACE/CAPSTONE/5. VIDEO/0. SOURCES VIDEO/video 2.mp4') # 0 stands for very first webcam attach
w,h = int(cap.get(3)), int(cap.get(4))
ret,imgF=cap.read(0)
imgF=Image.fromarray(imgF)
fps = cap.get(cv2.CAP_PROP_FPS)
print("fps: ", cap.get(cv2.CAP_PROP_FPS))

filename="/Users/acemitto/Desktop/Test YOLOv10.mp4"
codec=cv2.VideoWriter_fourcc('m','p','4','v')#fourcc stands for four character code
resolution=(1920,1080)
VideoFileOutput=cv2.VideoWriter(filename,codec,fps, resolution)

im_width, im_height = imgF.size

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
		results = model(frame, device="mps")[0]
		detections = sv.Detections.from_ultralytics(results)
		# {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter'
		# 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag'
		# 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard'
		# 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli'
		# 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop'
		# 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
		selected_classes = [1, 2, 3, 5, 7]
		detections = detections[np.isin(detections.class_id, selected_classes)]

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
			frame = zone_annotator.annotate(scene=frame)
			
		#print('yola')
		VideoFileOutput.write(frame)
		# frame_rs=cv2.resize(frame,(1020,647))
		# cv2image = cv2.cvtColor(frame_rs, cv2.COLOR_BGR2RGBA)
		# img = Image.fromarray(cv2image)
		# imgtk = ImageTk.PhotoImage(image=img)
		# display1.imgtk = imgtk #Shows frame for display 1
		# display1.configure(image=imgtk)
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


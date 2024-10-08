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
data_deque = {}
deepsort = None

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
    return img

wpod_net_path = "/Users/acemitto/Desktop/STUDY/ACE/STOCK/testData/Source/Traffic-Rule-Violation-Detection-System-master/wpod-net.json"
wpod_net = load_model(wpod_net_path)

def get_plate(image_path):
    vehicle = preprocess_image(image_path)
    results = model_lp(vehicle)
    detections = sv.Detections.from_yolov5(results)
    print(detections)
    if len(detections.xyxy) != 0:
        x = detections.xyxy[0]
        print(x)
        bimg=vehicle[int(x[1]):int(x[3]), int(x[0]):int(x[2])]
        return bimg
    else:
        return ''

def getLicensePlateNumber(url):
	path = url
	
	vehicle = get_plate(path)

	if vehicle != '' :
		if (len(vehicle)): #check if there is at least one license image
			# Scales, calculates absolute values, and converts the result to 8-bit.
			plate_image = cv2.convertScaleAbs(vehicle, alpha=(255.0))
			
			# convert to grayscale and blur the image
			gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
			blur = cv2.GaussianBlur(gray,(7,7),0)

		ocr_model = PaddleOCR(use_angle_cls=False, lang='en', show_log = False, use_gpu=True)
		result2 = ocr_model.ocr(blur)
		licensePlate = ""
		for i in range(len(result2[0])):
			if result2[0][i][1][0] != "" :
				licensePlate += result2[0][i][1][0]
			
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
			display5.configure(text=licensePlate)
	else :
		display5.configure(text="cant detected!")
		print("no plate!")
                
# calculate the speed of the object
def estimate_speed(location1, location2):
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    ppm = 19
    d_meters = d_pixels / ppm
    fps = 24
    speed = d_meters * fps * 3.6
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


def checkSpeed(kmh,bimg):
    if kmh >= 10:
        print(kmh)
        name='/Users/acemitto/Desktop/STUDY/ACE/STOCK/testData/Source/Traffic-Rule-Violation-Detection-System-master/Rule Breakers/Vi Pham '+str(time.time())+'.jpg'
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
        display3.configure(text=str(kmh)[:5]+'Km/h')

        cv2.imwrite(name,bimg)
        getLicensePlateNumber(name)
                                        
# draw the bounding box and the speed of the object
def UI_box(x, img, kmh, color=None, label=None, id=id, line_thickness=None):
	
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

		
		checkSpeed(kmh, bimg)


# draw the bounding box and the speed of the object
def draw_boxes(img, bbox, names,object_id, identities=None, offset=(0, 0)):
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


        color = compute_color_for_labels(object_id[i])
        obj_name = names[object_id[i]]
       
        label = '%s' % (obj_name)
        # add center to buffer
        data_deque[id].appendleft(center)
        # if data deque has more than two value, calculate speed
        if len(data_deque[id]) >= 2:
            object_speed = estimate_speed(data_deque[id][1], data_deque[id][0])
            speed_line_queue[id].append(object_speed)

        try:
            speed_check = int(sum(speed_line_queue[id])//len(speed_line_queue[id]))
            label = label + " " + str(sum(speed_line_queue[id])//len(speed_line_queue[id])) + "km/h"
        except:
            pass
        
        UI_box(box, img, speed_check, label=label, color=color, id=id ,line_thickness=1)
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
model_lp = yolov7.load("LP_detect_yolov7_500img.pt",trace=False)#),device="cuda:0")
print("model loaded")
print("Names: ", model.names)
print("Names: ", model_lp.names)

colors = sv.ColorPalette.default()
# initiate polygon zone
polygons = [ np.array([
    [340, 250],[900, 250],[1280, 600],[120, 600]
])]
polygons2 = [ np.array([
    [0, 350],[1280, 350],[1280, 720],[0, 720]
])]

cap=cv2.VideoCapture('/Users/acemitto/Desktop/STUDY/ACE/STOCK/testData/Source/Traffic-Rule-Violation-Detection-System-master/VietnamTraffic720p.mp4') # 0 stands for very first webcam attach
w,h = int(cap.get(3)), int(cap.get(4))
ret,imgF=cap.read(0)
imgF=Image.fromarray(imgF)
fps = cap.get(cv2.CAP_PROP_FPS)
print("fps: ", cap.get(cv2.CAP_PROP_FPS))

zones = [
    sv.PolygonZone(
        polygon=polygon, 
        frame_resolution_wh=(w,h)
    )
    for polygon
    in polygons2
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
        results = model(frame, size=500)
        detections = sv.Detections.from_yolov5(results)
        detections = (detections[(detections.class_id == 2)])
        # 2,3,5 = person, car, bus just detect these objects
        for zone, zone_annotator in zip(zones, zone_annotators):
        # annotate
            mask = zone.trigger(detections=detections)
            detections_filtered = detections[mask]

        xywh_bboxs = []
        confs = []
        oids = []
        outputs = []
        for xyxy, conf, cls in zip(detections_filtered.xyxy, detections_filtered.confidence, detections_filtered.class_id):
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
            
            frame = draw_boxes(frame, bbox_xyxy, model.names, object_id, identities)
        for zone, zone_annotator in zip(zones, zone_annotators):
            frame = zone_annotator.annotate(scene=frame)
            
        #print('yola')
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

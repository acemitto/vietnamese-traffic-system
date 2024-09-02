from ultralytics import YOLO
import torch
from numpy import random
from numpy import asarray
import math
import cv2
import time
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque
from shapely.geometry import Polygon
from shapely.geometry import Point
import numpy as np
import supervision as sv
import time
from datetime import datetime
import csv
import PIL.Image as Image
from PIL import ImageTk
from PIL import Image
import tkinter as tk
from wpodnet.wpodnet.backend import Predictor
from wpodnet.wpodnet.model import WPODNet
from paddleocr import PaddleOCR, draw_ocr # main OCR dependencies
from Database import KEYS, Database

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
speed_line_queue = {}
cross_line_queue = {}
data_deque = {}
deepsort = None

def init_tracker():
    global deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file('/Users/acemitto/Desktop/STUDY/ACE/CAPSTONE/vietnamese-traffic-system/Source/deep_sort_pytorch/configs/deep_sort.yaml')

    deepsort= DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                            max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

##################################### WPOD-NET #####################################

def load_model(path):
    try:
        # Prepare for the model
        device = torch.device('mps') if torch.backends.mps.is_available() else 'cpu'
        model = WPODNet()
        model.to(device)
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint)
        print('model wpod-net loaded successfully...')
        return model
    except Exception as e:
        print(e)

wpod_net_path = '/Users/acemitto/Desktop/STUDY/ACE/CAPSTONE/vietnamese-traffic-system/Source/wpodnet/weights/wpodnet.pth'
wpod_net = load_model(wpod_net_path)
predictor = Predictor(wpod_net)

#################################### processing ####################################

# convert the coordinates of the bounding box to the center coordinates of the bounding box
def xyxy_to_xywh(*xyxy):
    '''' Calculates the relative bounding box from absolute pixel values. '''
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def overlap2(rect1,rect2):
    p1 = Polygon([[rect1[0],rect1[1]], [rect1[2],rect1[1]],[rect1[2],rect1[3]],[rect1[0],rect1[3]]])
    p2 = Polygon([[rect2[0],rect2[1]], [rect2[2],rect2[1]],[rect2[2],rect2[3]],[rect2[0],rect2[3]]])
    if not p1.is_valid:
        p1 = p1.buffer(0)
    if not p2.is_valid:
        p2 = p1.buffer(0)

    return(p1.intersects(p2))

def preprocess_image(image_path):
    img = Image.open(image_path)
    return img

def compute_color_for_labels(label):
    '''
    Simple function that adds fixed color depending on the class
    '''
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


####################################################################################
##################################### FUNCTION #####################################

def get_plate(image_path, vehicle_type):
    if vehicle_type == 'motorcycle':
        Dmax = 600
        Dmin = 300
    else:
        Dmax = 600
        Dmin = 800

    vehicle = preprocess_image(image_path)
    prediction = predictor.predict(vehicle, scaling_ratio=1.1, dim_min=Dmin, dim_max=Dmax)

    return prediction

# sử dụng WPOD-NET để crop biển số và detect ký tự
def getLicensePlateNumber(url, obj_name):
    prediction = get_plate(url, obj_name)

    if (len(prediction)): # check if there is at least one license image
        # Scales, calculates absolute values, and converts the result to 8-bit.
        numpydata = asarray(prediction.warp())
        plate_image = cv2.convertScaleAbs(numpydata)
        # convert to grayscale and blur the image
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(7,7),0)

    licensePlate = ""

    ocr_model = PaddleOCR(use_angle_cls=False, lang='en', show_log = False, use_gpu=True)
    result2 = ocr_model.ocr(blur)

    for i in range(len(result2[0])):
        if result2[0][i][1][0] != "" :
            licensePlate += result2[0][i][1][0] + ' '
            print(result2[0][i][1][0])

    return licensePlate

# Calculation the speed of vehicles
def estimate_speed(timeCrossedTheGreenLight, timeCrossedTheBlueLight):
    d_time = (timeCrossedTheGreenLight - timeCrossedTheBlueLight) /1000
    d_meters = 3
    speed = d_meters / d_time * 3.6
    return int(speed)

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
        if data_deque[1][1] >= yl2 and data_deque[0][1] < yl2:
            return True
        else :
            return False
    elif  direction_obj == 'down':
        if data_deque[1][1] < yl2 and data_deque[0][1] >= yl2:
            return True
        else :
            return False

# kiểm tra sai làn    
def checkWrongLane(data_deque, direction, obj_name):
    if len(polygons) == 4:
        polygon_1 = Polygon(polygons[0]) # lane 1
        # không cần lane 2 vì là lane chung
        polygon_3 = Polygon(polygons[2]) # lane 2
        polygon_4 = Polygon(polygons[3]) # lane 4
        polygon_lane = Polygon(polygons_lane[0]) # khung toàn bộ đường
        point = Point(data_deque[0])

        if direction == 'down':
            if polygon_lane.contains(point):
                return 'Wrong Direction'
        
        if direction == 'up':
            if obj_name == 'motorcycle' or obj_name == 'bicycle':
                    if polygon_1.contains(point):
                        return 'warning: Wrong Lane'
                    
            elif obj_name == 'car' or obj_name == 'truck' or obj_name == 'bus':
                if polygon_3.contains(point):
                    return 'warning: Wrong Lane'
                if polygon_4.contains(point):
                    return 'warning: Wrong Lane'
            
        elif direction == 'stay':
            if polygon_4.contains(point):
                    return 'Warning: Cannot Parking Here'
        else:
            return False
    elif len(polygons) == 3 :
        polygon_1 = Polygon(polygons[0])
        polygon_2 = Polygon(polygons[1])
        polygon_3 = Polygon(polygons[2])
        polygon_lane = Polygon(polygons_lane[0])
        point = Point(data_deque[0])

        if direction == 'down':
            if obj_name == 'motorcycle':
                if polygon_lane.contains(point):
                    return 'Wrong Direction'
        
        if direction == 'up':
            if obj_name == 'motorcycle' or obj_name == 'bicycle':
                    if polygon_1.contains(point):
                        return 'warning: Wrong Lane'
                    
            elif obj_name == 'car' or obj_name == 'truck' or obj_name == 'bus':
                if polygon_3.contains(point):
                    return 'warning: Wrong Lane'
                if polygon_3.contains(point):
                    return 'warning: Wrong Lane'
            
        elif direction == 'stay':
            if polygon_3.contains(point):
                    return 'Warning: Cannot Parking Here'
        else:
            return False
    else :
        polygon_1 = Polygon(polygons[0])
        polygon_2 = Polygon(polygons[1])
        polygon_lane = Polygon(polygons_lane[0])
        point = Point(data_deque[0])

        if direction == 'down':
            if polygon_2.contains(point):
                return 'Wrong Direction'
        
        if direction == 'up':
            if polygon_1.contains(point):
                return 'Wrong Direction'
        else:
            return False

def checkHelmet(box_obj, bbox, object_id):
    count = 0
    for i in range(len(bbox)):
        if object_id[i] == 1:
            if overlap2(bbox[i], box_obj):
                count = count + 1
    if count == 0:
        return 'No Helmet'
    else:
        return False
    
def checkOverPerson(box_obj, bbox, object_id):
    count = 0
    for i in range(len(bbox)):
        if object_id[i] == 0:
            if overlap2(bbox[i], box_obj):
                count = count + 1
    if count >= 3:
        return 'Over Person'
    else:
        return False
    
# kiểm tra vận tốc
def  checkSpeed(kmh,bimg, obj_name, data_deque):
    if data_deque[1][1]>yl2 and data_deque[0][1]<yl2:
        if kmh >= 50: # giới hạn vận tốc
            name='/Users/acemitto/Desktop/STUDY/ACE/CAPSTONE/vietnamese-traffic-system/Source/Traffic-Rule-Violation-Detection-System-master/rule_breakers/Vi Pham '+str(time.time())+'.jpg'
            
            h, w = bimg.shape[:2]
            frame2=bimg
            img2 = Image.fromarray(frame2)
            w,h=img2.size
            asprto=w/h
            frame2=cv2.resize(frame2,(250,int(250/asprto)))
            cv2image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGBA)
            img2 = Image.fromarray(cv2image2)
            imgtk2 = ImageTk.PhotoImage(image=img2)

            cv2.imwrite(name,bimg)
            license_plate_queue = getLicensePlateNumber(name, obj_name)
            
            # getDataViolation(kmh, obj_name, name, license_plate_queue)
        

####################################################################################
################################# Draw Bouncingbox #################################
# draw the bounding box and the speed of the object
def UI_box(x, ori_img, img, kmh, obj_name, data_deque, wrong_lane, no_helmet, over_person, color=None, label=None, id=id, line_thickness=None):
    tl = line_thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    w, h = int(x[2] - x[0]), int(x[3] - x[1])
    cx, cy = int(x[0] + w / 2), int(x[1] + h / 2)

    bimg=ori_img[int(x[1]):int(x[3]), int(x[0]):int(x[2])]
    cv2.circle(img, (cx, cy), 2, (0,0,255), cv2.FILLED)
    img = cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        
        # img = cv2.rectangle(img, (c1[0], c1[1] - t_size[1] - 3), (c1[0] + t_size[0], c1[1] - 2), color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)    

        checkSpeed(kmh, bimg, obj_name, data_deque)

    if wrong_lane:
        t_size = cv2.getTextSize(wrong_lane, 0, fontScale=1, thickness=tf)[0]
        img = cv2.rectangle(img, (c1[0], c1[1] - t_size[1] - 3), (c1[0] + t_size[0], c1[1]), [0,0,255], -1, cv2.LINE_AA)  # filled
        cv2.putText(img, wrong_lane, (c1[0], c1[1] - 2), 0, 1, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)

    if no_helmet:
        t_size = cv2.getTextSize(no_helmet, 0, fontScale=1, thickness=tf)[0]
        img = cv2.rectangle(img, (c1[0], c1[1] - t_size[1] - 3), (c1[0] + t_size[0], c1[1]), [0,0,255], -1, cv2.LINE_AA)  # filled
        cv2.putText(img, no_helmet, (c1[0], c1[1] - 2), 0, 1, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)

    if over_person:
        t_size = cv2.getTextSize(over_person, 0, fontScale=1, thickness=tf)[0]
        img = cv2.rectangle(img, (c1[0], c1[1] - t_size[1] - 3), (c1[0] + t_size[0], c1[1]), [0,0,255], -1, cv2.LINE_AA)  # filled
        cv2.putText(img, over_person, (c1[0], c1[1] - 2), 0, 1, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)

# draw the bounding box and the speed of the object
def draw_boxes(ori_img, img, bbox, names, object_id, cur_time, identities=None, offset=(0, 0)):
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
        if object_id[i] == 1:
            obj_name = 'helmet'
        

        label = '{}{:d}'.format('', id) + ': '+ '%s' % (obj_name)
        # add center to buffer
        data_deque[id].appendleft(center)
        # if data deque has more than two value, calculate speed

        if len(data_deque[id]) >= 2:
            wrong_lane = ''
            no_helmet = ''
            over_person = ''
            direction_obj = ''
            
            if len(data_deque[id]) >= 6:
                if object_id[i] != 1:
                    direction_obj = getDirection(data_deque[id])
                    label = label + ' ' + str(direction_obj)
                    wrong_lane = checkWrongLane(data_deque[id], direction_obj, obj_name)
                    
            if object_id[i] == 0:
                no_helmet = checkHelmet(box, bbox, object_id)
                over_person = checkOverPerson(box, bbox, object_id)

            if object_id[i] != 1 and object_id[i] != 0:
                if checkCrossedTheBlueLine(data_deque[id], direction_obj):
                    if direction_obj == 'up':
                        cross_line_queue[id].append(cur_time)
                    elif direction_obj == 'down':
                        if int(sum(cross_line_queue[id])) != 0:
                            object_speed = estimate_speed(cur_time, sum(cross_line_queue[id]))
                            speed_line_queue[id].append(object_speed)
                            # getDataCrossedTheLight(id, object_speed, obj_name)

                if checkCrossedTheGreenLine(data_deque[id], direction_obj):
                    if direction_obj == 'up':
                        if int(sum(cross_line_queue[id])) != 0:
                            object_speed = estimate_speed(cur_time, sum(cross_line_queue[id]))
                            speed_line_queue[id].append(object_speed)
                            # getDataCrossedTheLight(id, object_speed, obj_name)
                    elif direction_obj == 'down':
                        cross_line_queue[id].append(cur_time)

            if  int(sum(speed_line_queue[id])) != 0:
                speed_check = int(sum(speed_line_queue[id])//len(speed_line_queue[id]))
                label = label + ' ' + str(int(sum(speed_line_queue[id]))) + 'km/h'
            UI_box(box, ori_img, img, speed_check, obj_name, data_deque[id], wrong_lane, no_helmet, over_person, label=label, color=color, id=id ,line_thickness=3)
        
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

def insertDb(self):
    db = Database.getInstance()
    db.deleteViolation(carid=self.data[KEYS.CARID], ruleid=self.data[KEYS.RULEID], time=self.data[KEYS.TIME])
    self.destroy()

init_tracker()

# load yolov9 model
# pip install yolov7detect
# Load a model
# model = YOLO('yolov9e.yaml')  # build a new model from scratch
model = YOLO('yolov9e.pt')  # load a pretrained model (recommended for training)
print('model YOLO loaded successfully...')

model_helmet = YOLO('helmetYolov8.pt')  # load a pretrained model (recommended for training)
print('model helmet loaded successfully...')

colors = sv.ColorPalette.default()
# initiate polygon zone
# [0, 0],[1920, 0],[1920, 1080],[0, 1080]

polygons_lane = []

polygons = []

cap=''
w,h = int(0), int(0)

xl1=0
xl2=0
yl1=0 
yl2=0

xj1=0
xj2=0
yj1=0
yj2=0
zones_lane = []
zone_annotators_lane = []
box_annotators_lane = []
zones = []
zone_annotators = []
box_annotators = []
VideoFileOutput = ''
cnttt=0
vehicles=[]


class MainProcessor:
    def __init__(url, camera_id):
        print('Running')

    def preCode(self, url, pl_lane, pl, yl, yj):
        global polygons_lane
        global polygons
        global cap
        global w,h
        global xl1,xl2,yl1,yl2
        global xj1,xj2,yj1,yj2
        global zones_lane, zone_annotators_lane, box_annotators_lane
        global zones, zone_annotators, box_annotators
        global VideoFileOutput
        
        polygons_lane = pl_lane

        polygons = pl

        cap=cv2.VideoCapture(url) # 0 stands for very first webcam attach
        w,h = int(cap.get(3)), int(cap.get(4))
        ret,imgF=cap.read(0)
        imgF=Image.fromarray(imgF)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print('fps: ', cap.get(cv2.CAP_PROP_FPS))

        filename='/Users/acemitto/Desktop/STUDY/ACE/CAPSTONE/vietnamese-traffic-system/Source/videos/Outpput/YOLOv9 Multi Lane Wrong Lane2.mp4'
        codec=cv2.VideoWriter_fourcc('m','p','4','v')#fourcc stands for four character code
        resolution=(3840,2160)
        VideoFileOutput=cv2.VideoWriter(filename,codec,fps, resolution)

        im_width, im_height = imgF.size
        xl1=0
        xl2=im_width-1
        yl1=int(yl) 
        yl2=yl1

        xj1=0
        xj2=im_width-1
        yj1=int(yj)
        yj2=yj1

        zones_lane = [
            sv.PolygonZone(
                polygon=polygon, 
                frame_resolution_wh=(w,h)
            )
            for polygon
            in polygons_lane
        ]

        zone_annotators_lane = [
            sv.PolygonZoneAnnotator(
                zone=zone, 
                color=colors.by_idx(index), 
                thickness=2
            )
            for index, zone in enumerate(zones_lane)
        ]

        box_annotators_lane = [
            sv.BoxAnnotator(
                color=colors.by_idx(index), 
                thickness=1, 
                text_thickness=1, 
                text_scale=0.5
                )
            for index
            in range(len(polygons_lane))
        ]

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
    
    def getProcessedImage(self):
        if True:
            fTime=time.time()
            ret, frame = cap.read(0)
            ori_frame = frame.copy()
            # detect
            results = model(frame, device='mps')[0]
            detections = sv.Detections.from_ultralytics(results)
            # {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter'
            # 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag'
            # 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard'
            # 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli'
            # 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop'
            # 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
            selected_classes = [0, 2, 3, 5, 7, 9]
            detections = detections[np.isin(detections.class_id, selected_classes)]

            results_helmet = model_helmet(frame, device='mps')[0]
            detections_helmet = sv.Detections.from_ultralytics(results_helmet)
            
            for zone, zone_annotator in zip(zones_lane, zone_annotators_lane):
                # annotate
                mask = zone.trigger(detections=detections)
                detections_filtered = detections[mask]
                mask_hm = zone.trigger(detections=detections_helmet)
                detections_filtered_hm = detections_helmet[mask_hm]

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

            for xyxy2, conf2, cls in zip(detections_filtered_hm.xyxy, detections_filtered_hm.confidence, detections_filtered_hm.class_id):
                x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy2)
                xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                xywh_bboxs.append(xywh_obj)
                confs.append([conf2.item()])
                oids.append(int(cls))

            xywhs = torch.Tensor(xywh_bboxs)
            confss = torch.Tensor(confs)
            if xywhs.nelement() > 0:
                outputs = deepsort.update(xywhs, confss, oids, frame)

            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -2]
                object_id = outputs[:, -1]

                frame = draw_boxes(ori_frame, frame, bbox_xyxy, model.names, object_id, cap.get(cv2.CAP_PROP_POS_MSEC), identities)

            for zone, zone_annotator in zip(zones, zone_annotators):
                frame = zone_annotator.annotate(scene=frame)

            #print('yola')
            cv2.line(frame, (int(xl1),int(yl1)), (int(xl2),int(yl2)), (0,255,0),2)
            cv2.line(frame, (int(xj1),int(yj1)), (int(xj2),int(yj2)), (255,0,0),2 )
            VideoFileOutput.write(frame)
            frame_rs=cv2.resize(frame,(1020,647))
            cv2image = cv2.cvtColor(frame_rs, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)

        return frame_rs
    

    def setLight(self, color):
        self.processor.light = color

    def getLight(self):
        return self.processor.light
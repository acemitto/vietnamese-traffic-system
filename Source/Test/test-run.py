from ultralytics import YOLO
import torch
from numpy import random
from numpy import asarray
import math
import cv2
import time
from utils.deep_sort_pytorch.utils.parser import get_config
from utils.deep_sort_pytorch.deep_sort import DeepSort
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
import tkinter as tk
from utils.wpodnet.wpodnet.backend import Predictor
from utils.wpodnet.wpodnet.model import WPODNet
from paddleocr import PaddleOCR, draw_ocr # main OCR dependencies
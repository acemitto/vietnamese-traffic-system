from ultralytics import YOLO
from os.path import splitext
from keras.models import model_from_json

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
speed_line_queue = {}
cross_line_queue = {}
data_deque = {}
deepsort = None

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

wpod_net_path = "/Users/acemitto/Desktop/STUDY/ACE/CAPSTONE/vietnamese-traffic-system/Source/Traffic-Rule-Violation-Detection-System-master/wpod-net.json"
wpod_net = load_model(wpod_net_path)


# load yolov8 model
# pip install yolov7detect
# Load a model
model = YOLO("yolov8x.yaml")  # build a new model from scratch
model = YOLO("yolov8x.pt")  # load a pretrained model (recommended for training)

print("model loaded")
print("Names: ", model.names)

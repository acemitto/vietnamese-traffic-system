import cv2
import torch
from paddleocr import PaddleOCR # main OCR dependencies
from wpodnet.backend import Predictor
from wpodnet.model import WPODNet
from numpy import asarray
from PIL import Image
import time
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    return img


# Prepare for the model
device = torch.device('mps') if torch.backends.mps.is_available() else 'cpu'
model = WPODNet()
model.to(device)

checkpoint = torch.load('/Users/acemitto/Desktop/STUDY/ACE/CAPSTONE/vietnamese-traffic-system/Source/wpodnet/weights/wpodnet.pth')
model.load_state_dict(checkpoint)

predictor = Predictor(model)

path = '/Users/acemitto/Desktop/STUDY/ACE/CAPSTONE/vietnamese-traffic-system/Source/rule_breakers/Vi Pham 1698571917.868161.jpg'
streamer = Image.open(path)

Dmax = 600
Dmin = 800

a = time.time()
prediction = predictor.predict(streamer, scaling_ratio=1.1, dim_min=Dmin, dim_max=Dmax)
print(time.time() - a)
print('Bounds', prediction.bounds.tolist())
print('Confidence', prediction.confidence)

numpydata = asarray(prediction.warp())
plate_image = cv2.convertScaleAbs(numpydata)
gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(7,7),0)


# if save_annotated:
#     annotated_path = save_annotated / Path(image.filename).name
#     annotated = prediction.annotate()
#     annotated.save(annotated_path)
#     print(f'Saved the annotated image at {annotated_path}')

# if save_warped:
#     warped_path = save_warped / Path(image.filename).name
#     warped = prediction.warp()
#     warped.save(warped_path)\
#     print(f'Saved the warped image at {warped_path}')

licensePlate = ""

ocr = PaddleOCR(use_gpu=False, use_angle_cls=True, lang='en', show_log=True)
result = ocr.ocr(blur, cls=True)

for i in range(len(result[0])):
    if result[0][i][1][0] != "" :
        licensePlate += result[0][i][1][0] + ' '
        print(result[0][i][1][0])



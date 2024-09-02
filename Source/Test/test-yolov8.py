from ultralytics import YOLO

# Load a model
# build a new model from scratch
model = YOLO("yolov8x.yaml")

# load a pretrained model (recommended for training)
model = YOLO("yolov8x.pt") 

# Use the model
model.train(data='coco.yaml', epochs=500, imgsz=640, device='mps')

# Display model information (optional)
model.info()

path = model.export(format="onnx")  # export the model to ONNX format
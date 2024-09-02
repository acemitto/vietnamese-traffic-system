from ultralytics import YOLO

# Load a model
# Build a YOLOv9e model from scratch
model = YOLO('yolov9e.yaml')

# load a pretrained model (recommended for training)
model = YOLO('yolov9e.pt')

# Use the model
model.train(data='coco.yaml', epochs=500, imgsz=640, device='mps')

# Display model information (optional)
model.info()

path = model.export(format="onnx")  # export the model to ONNX format
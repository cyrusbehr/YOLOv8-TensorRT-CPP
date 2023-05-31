from ultralytics import YOLO

model = YOLO("../models/yolov8s.pt")
model.fuse()
model.info(verbose=False)  # Print model information
model.export(format="onnx", opset=12)
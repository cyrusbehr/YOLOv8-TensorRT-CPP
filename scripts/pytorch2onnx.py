from ultralytics import YOLO

model = YOLO("../models/yolov8x.pt") 
model.export(format="onnx", imgsz=[640,640])
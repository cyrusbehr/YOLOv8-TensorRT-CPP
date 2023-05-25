# YOLOv8-TensorRT-CPP
YOLOv8 TensorRT C++ Implementation

### Installation

git clone --recursive

* Important to use recursive as we use submodules! 

### Converting Model from Onnx

Download the model from this repo: https://github.com/ultralytics/ultralytics#models

Use the following python code to export the model to onnx:

```python
from ultralytics import YOLO

model = YOLO("yolov8m.pt") 
model.export(format="onnx", imgsz=[640,640])
```
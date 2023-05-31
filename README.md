# YOLOv8-TensorRT-CPP
YOLOv8 TensorRT C++ Implementation

### Prereqs
- Install cuda
- Install TRT, and explain where in the code to set TRT dir
- Install OpenCV with CUDA support, which can be done here. 

### Installation

git clone --recursive

* Important to use recursive as we use submodules! 

### Converting Model from Onnx

pip3 install ultralytics

Download the model from this repo: https://github.com/ultralytics/ultralytics#models

Use the script in /scripts to convert your model. 

- Then change the model name. 
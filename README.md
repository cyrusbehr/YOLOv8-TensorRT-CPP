[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/cyrusbehr/YOLOv8-TensorRT-CPP">
    <img width="70%" src="assets/yolov8.gif" alt="logo">
  </a>
  <h3 align="center">YoloV8 TensorRT CPP</h3>
</p>


### Getting Started
This project demonstrates how to use the TensorRT C++ API to run GPU inference for YoloV8. 
It makes use of my other project [tensorrt-cpp-api](https://github.com/cyrusbehr/tensorrt-cpp-api) to run inference behind the scene, so make sure you are familiar with that project.

### Prerequisites
- Tested and working on Ubuntu 20.04
- Install CUDA, instructions [here](https://developer.nvidia.com/cuda-11.2.0-download-archive).
    - Recommended >= 11.2
- Install cudnn, instructions [here](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#download).
    - Recommended >= 8
- `sudo apt install build-essential`
- `sudo apt install python3-pip`
- `pip3 install cmake`
- Install OpenCV with cuda support. Instructions can be found [here](https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7).
- Download TensorRT 8 from [here](https://developer.nvidia.com/nvidia-tensorrt-8x-download).
    - Recommended >= 8.2
- Extract, and then navigate to the `CMakeLists.txt` file and replace the `TODO` with the path to your TensorRT installation.


### Installation
- `git clone https://github.com/cyrusbehr/YOLOv8-TensorRT-CPP --recursive`
- **Note:** Be sure to use the `--recursive` flag as this repo makes use of git submodules. 

### Converting Model from PyTorch to ONNX
- Navigate to the [official YoloV8 repository](https://github.com/ultralytics/ultralytics) and download your desired version of the model (ex. YOLOv8m).
- `pip3 install ultralytics`
- Navigate to the `scripts/` directory and modify this line so that it points to your downloaded model: `model = YOLO("../models/yolov8m.pt")`.
- `python3 pytorch2onnx.py`
- After running this command, you should successfully have converted from PyTorch to ONNX. 

### Building the Project
- `mkdir build`
- `cd build`
- `cmake ..`
- `make -j`

### Running the Executables
- *Note*: the first time you run any of the scripts, it may take quite a long time (5 mins+) as TensorRT must generate an optimized TensorRT engine file from the onnx model. This is then saved to disk and loaded on subsequent runs.
- To run the benchmarking script, run: `./benchmark /path/to/your/onnx/model.onnx`
- To run inference on an image and save the annotated image to disk run: `./detect_object_image /path/to/your/onnx/model.onnx /path/to/your/image.jpg`
  - You can use the images in the `images/` directory for testing
- To run inference using your webcam and display the results in real time, run: `./detect_object_video /path/to/your/onnx/model.onnx`
  - To change the video source, navigate to `src/object_detection_video_streaming.cpp` and change this line to your specific video source: ` cap.open(0);`.
  - The video source can be an int or a string (ex. "/dev/video4" or an RTSP url).

### How to debug
- If you have issues creating the TensorRT engine file from the onnx model, navigate to `libs/tensorrt-cpp-api/src/engine.cpp` and change the log level by changing the severity level to `kVERBOSE` and rebuild and rerun. This should give you more information on where exactly the build process is failing.

### Show your appreciation
If this project was helpful to you, I would appreicate if you could give it a star. That will encourage me to ensure it's up to date and solve issues quickly.


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[stars-shield]: https://img.shields.io/github/stars/cyrusbehr/YOLOv8-TensorRT-CPP.svg?style=flat-square
[stars-url]: https://github.com/cyrusbehr/YOLOv8-TensorRT-CPP/stargazers
[issues-shield]: https://img.shields.io/github/issues/cyrusbehr/YOLOv8-TensorRT-CPP.svg?style=flat-square
[issues-url]: https://github.com/cyrusbehr/YOLOv8-TensorRT-CPP/issues
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/cyrus-behroozi/

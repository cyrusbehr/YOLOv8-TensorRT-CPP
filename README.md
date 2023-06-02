[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/cyrusbehr/YOLOv8-TensorRT-CPP">
    <img width="95%" src="assets/yolov8.png" alt="logo">
  </a>
  <h3 align="center">YoloV8 TensorRT CPP</h3>
  <p align="center">
    <b>
    A C++ Implementation of YoloV8 using TensorRT
    </b>

[//]: # (    <br />)
[//]: # (    Supports models with single / multiple inputs and single / multiple outputs with batching.)
[//]: # (    <br />)
[//]: # (    <br />)
[//]: # (    <a href="https://www.youtube.com/watch?v=kPJ9uDduxOs">Project Overview Video</a>)
[//]: # (    .)
[//]: # (    <a href="https://youtu.be/Z0n5aLmcRHQ">Check of </a>)

  </p>
</p>


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


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[stars-shield]: https://img.shields.io/github/stars/cyrusbehr/YOLOv8-TensorRT-CPP.svg?style=flat-square
[stars-url]: https://github.com/cyrusbehr/YOLOv8-TensorRT-CPP/stargazers
[issues-shield]: https://img.shields.io/github/issues/cyrusbehr/YOLOv8-TensorRT-CPP.svg?style=flat-square
[issues-url]: https://github.com/cyrusbehr/YOLOv8-TensorRT-CPP/issues
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/cyrus-behroozi/

#include <opencv2/opencv.hpp>
#include <chrono>
#include <fstream>
#include "engine.h"

typedef std::chrono::high_resolution_clock Clock;

inline bool doesFileExist (const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Error: Must specify input image" << std::endl;
        std::cout << "Usage: " << argv[0] << " /path/to/your/image.jpg" << std::endl;
        return -1;
    }

    if (argc > 2) {
        std::cout << "Error: Too many arguments provided" << std::endl;
        std::cout << "Usage: " << argv[0] << " /path/to/your/image.jpg" << std::endl;
    }

    const std::string inputImgPath = argv[1];
    if (!doesFileExist(inputImgPath)) {
        std::cout << "Error: The input file does not exist" << std::endl;
        return -1;
    }

    // Specify options for GPU inference
    Options options;
    // This particular model only supports a fixed batch size of 1
    options.doesSupportDynamicBatchSize = false;
    options.optBatchSize = 1;
    options.maxBatchSize = 1;
    options.maxWorkspaceSize = 2000000000;

    // Use FP16 precision to speed up inference
    options.precision = Precision::FP16;

    Engine engine(options);

    // Specify the model we want to load
    const std::string onnxModelPath = "../models/yolov8x.onnx";

    // Build the onnx model into a TensorRT engine file
    // If the engine file already exists, this function will return immediately
    // The engine file is rebuilt any time the above Options are changed. 
    auto succ = engine.build(onnxModelPath);
    if (!succ) {
        std::cout << "Error: Unable to build the TensorRT engine" << std::endl;
        std::cout << "Try increasing TensorRT log severity to kVERBOSE (in /libs/tensorrt-cpp-api/engine.cpp)" << std::endl;
        return -1;
    }

    // Load the TensorRT engine file
    succ = engine.loadNetwork();
    if (!succ) {
        std::cout << "Error: Unable to load TensorRT engine weights" << std::endl;
        return -1;
    }


    return 0;
}

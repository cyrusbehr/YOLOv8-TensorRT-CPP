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

    const std::string inputImage = argv[1];
    if (!doesFileExist(inputImage)) {
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

    size_t batchSize = options.optBatchSize;

    // Read the input image
    auto cpuImg = cv::imread(inputImage);
    if (cpuImg.empty()) {
        std::cout << "Error: Unable to read image at path: " << inputImage << std::endl;
        return -1;
    }

    // The model expects RGB input
    cv::cvtColor(cpuImg, cpuImg, cv::COLOR_BGR2RGB);

    // Upload to GPU memory
    cv::cuda::GpuMat img;
    img.upload(cpuImg);

    // Populate the input vectors
    const auto& inputDims = engine.getInputDims();
    std::vector<std::vector<cv::cuda::GpuMat>> inputs;

    for (const auto & inputDim : inputDims) {
        std::vector<cv::cuda::GpuMat> input;
        for (size_t j = 0; j < batchSize; ++j) {
            // Resize to the model expected input size while maintaining the aspect ratio with the use of padding
            if (inputDim.d[2] != inputDim.d[1]) {
                std::cout << "Error: Resize method expected model with h=w"
            }
            auto resized = Engine::resizeKeepAspectRatioPadRightBottom(img, inputDim.d[2])
            cv::cuda::resize(img, resized, cv::Size(inputDim.d[2], inputDim.d[1])); // TRT dims are (height, width) whereas OpenCV is (width, height)
            input.emplace_back(std::move(resized));
        }
        inputs.emplace_back(std::move(input));
    }

    return 0;
}

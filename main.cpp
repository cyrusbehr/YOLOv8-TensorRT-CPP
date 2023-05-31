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
    const std::string onnxModelPath = "../models/yolov8m.onnx";

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
            auto resized = Engine::resizeKeepAspectRatioPadRightBottom(img, inputDim.d[1], inputDim.d[2]);
            input.emplace_back(std::move(resized));
        }
        inputs.emplace_back(std::move(input));
    }

    // Define our preprocessing code
    // YoloV8 model expects values between [0.f, 1.f] so we use the following params
    std::array<float, 3> subVals {0.f, 0.f, 0.f};
    std::array<float, 3> divVals {1.f, 1.f, 1.f};
    bool normalize = true;

    // Run inference 10 times to warm up the engine
    for (int i = 0; i < 10; ++i) {
        std::vector<std::vector<std::vector<float>>> featureVectors;
        succ = engine.runInference(inputs, featureVectors, subVals, divVals, normalize);
        if (!succ) {
            std::cout << "Error: Unable to run inference" << std::endl;
            return -1;
        }
    }

    // Now run the benchmark
    auto t1 = Clock::now();
    size_t numIts = 100;
    for (size_t i = 0; i < numIts; ++i) {
        std::vector<std::vector<std::vector<float>>> featureVectors;
        succ = engine.runInference(inputs, featureVectors, subVals, divVals, normalize);
        if (!succ) {
            std::cout << "Error: Unable to run inference" << std::endl;
            return -1;
        }

        // TODO Cyrus: Include post processing here, and include it in the benchmark
    }
    auto t2 = Clock::now();
    double totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    std::cout << "Average time per inference: " << totalTime / numIts / static_cast<float>(inputs[0].size()) <<
              " ms, for batch size of: " << inputs[0].size() << std::endl;



    return 0;
}

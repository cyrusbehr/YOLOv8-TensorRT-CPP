#include <opencv2/opencv.hpp>
#include <chrono>
#include "engine.h"

typedef std::chrono::high_resolution_clock Clock;

int main() {
    // Specify options for GPU inference
    Options options;
    // This particular model only supports a fixed batch size of 1
    options.doesSupportDynamicBatchSize = false;
    options.optBatchSize = 1;
    options.maxBatchSize = 1;

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
        throw std::runtime_error("Unable to build the TensorRT engine");
    }

    // Load the TensorRT engine file
    succ = engine.loadNetwork();


    return 0;
}

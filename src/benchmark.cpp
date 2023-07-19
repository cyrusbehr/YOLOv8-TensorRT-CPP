#include "yolov8.h"
#include <opencv2/cudaimgproc.hpp>

// Benchmarks the specified model
int main(int argc, char *argv[]) {
    // Parse the command line arguments
    // Must pass the model path as a command line argument to the executable
    if (argc < 2) {
        std::cout << "Error: Must specify the model path" << std::endl;
        std::cout << "Usage: " << argv[0] << " /path/to/onnx/model.onnx" << std::endl;
        return -1;
    }

    if (argc > 3) {
        std::cout << "Error: Too many arguments provided" << std::endl;
        std::cout << "Usage: " << argv[0] << " /path/to/onnx/model.onnx" << std::endl;
    }

    // Ensure the onnx model exists
    const std::string onnxModelPath = argv[1];
    if (!doesFileExist(onnxModelPath)) {
        std::cout << "Error: Unable to find file at path: " << onnxModelPath << std::endl;
        return -1;
    }

    // Load our benchmarking image
    const std::string inputImage = "../images/640_640.jpg";
    if (!doesFileExist(inputImage)) {
        std::cout << "Error: Unable to find file at path: " << inputImage << std::endl;
        return -1;
    }

    // Create our YoloV8 engine
    // Use default probability threshold, nms threshold, and top k
    YoloV8 yoloV8(onnxModelPath);

    // Read the input image
    auto img = cv::imread(inputImage);
    if (img.empty()) {
        std::cout << "Error: Unable to read image at path: " << inputImage << std::endl;
        return -1;
    }

    // Upload the image to GPU memory
    cv::cuda::GpuMat gpuImg;
    gpuImg.upload(img);

    // Convert from BGR to RGB
    cv::cuda::cvtColor(gpuImg, gpuImg, cv::COLOR_BGR2RGB);

    std::cout << "Benchmarking network with image of size: (" << gpuImg.cols << ", " << gpuImg.rows << ")" << std::endl;

    // Warm up the network
    std::cout << "Warming up the network..." << std::endl;
    size_t numIts = 50;
    for (size_t i = 0; i < numIts; ++i) {
        const auto objects = yoloV8.detectObjects(gpuImg);
    }

    // Draw the bounding boxes on the image
    numIts = 200;
    std::cout << "Warmup done. Running benchmarks (" << numIts << " iterations)..." << std::endl;
    preciseStopwatch stopwatch;
    for (size_t i = 0; i < numIts; ++i) {
        const auto objects = yoloV8.detectObjects(gpuImg);
    }

    auto totalElapsedTimeMs = stopwatch.elapsedTime<float, std::chrono::milliseconds>();
    auto avgElapsedTimeMs = totalElapsedTimeMs / numIts;

    std::cout << "Benchmarking complete!" << std::endl;
    std::cout << "======================" << std::endl;
    std::cout << "Avg time: " << std::endl;
    std::cout << avgElapsedTimeMs << "ms\n" << std::endl;
    std::cout << "Avg FPS: " << std::endl;
    std::cout << static_cast<int>(1000 / avgElapsedTimeMs) << " fps" << std::endl;
    std::cout << "======================" << std::endl;

    return 0;
}

#include "yolov8.h"
#include "cmd_line_util.h"
#include <opencv2/cudaimgproc.hpp>


// Benchmarks the specified model
int main(int argc, char *argv[]) {
    YoloV8Config config;
    std::string onnxModelPath;
    std::string inputImage;

    // Parse the command line arguments
	if (!parseArguments(argc, argv, config, onnxModelPath, inputImage)) {
		return -1;
    }

    // Create the YoloV8 engine
    YoloV8 yoloV8(onnxModelPath, config);

    // Read the input image
    auto img = cv::imread(inputImage);
    if (img.empty()) {
        std::cout << "Error: Unable to read image at path: " << inputImage << std::endl;
        return -1;
    }

    // Upload the image to GPU memory
    cv::cuda::GpuMat gpuImg;
    gpuImg.upload(img);

    std::cout << "Benchmarking network with image of size: (" << gpuImg.cols << ", " << gpuImg.rows << ")" << std::endl;

    // Warm up the network
    std::cout << "Warming up the network..." << std::endl;
    size_t numIts = 50;
    for (size_t i = 0; i < numIts; ++i) {
        const auto objects = yoloV8.detectObjects(gpuImg);
    }

    // Draw the bounding boxes on the image
    numIts = 2000;
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
    std::cout << avgElapsedTimeMs << " ms\n" << std::endl;
    std::cout << "Avg FPS: " << std::endl;
    std::cout << static_cast<int>(1000 / avgElapsedTimeMs) << " fps" << std::endl;
    std::cout << "======================" << std::endl;

    return 0;
}
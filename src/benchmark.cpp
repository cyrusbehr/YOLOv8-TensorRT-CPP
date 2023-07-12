#include <chrono>
typedef std::chrono::high_resolution_clock Clock;

#include "yolov8.h"

// Utility Timer
template <typename Clock = std::chrono::high_resolution_clock>
class Stopwatch
{
    typename Clock::time_point start_point;
public:
    Stopwatch() :start_point(Clock::now()){}

    // Returns elapsed time
    template <typename Rep = typename Clock::duration::rep, typename Units = typename Clock::duration>
    Rep elapsedTime() const {
        std::atomic_thread_fence(std::memory_order_relaxed);
        auto counted_time = std::chrono::duration_cast<Units>(Clock::now() - start_point).count();
        std::atomic_thread_fence(std::memory_order_relaxed);
        return static_cast<Rep>(counted_time);
    }
};

using preciseStopwatch = Stopwatch<>;
using systemStopwatch = Stopwatch<std::chrono::system_clock>;
using monotonicStopwatch = Stopwatch<std::chrono::steady_clock>;


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
    const std::string inputImage = "../images/cars.jpg";
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

    std::cout << "Benchmarking network with image of size: (" << img.cols << ", " << img.rows << ")" << std::endl;

    // Warm up the network
    std::cout << "Warming up the network..." << std::endl;
    size_t numIts = 50;
    for (size_t i = 0; i < numIts; ++i) {
        const auto objects = yoloV8.detectObjects(img);
    }

    // Draw the bounding boxes on the image
    numIts = 200;
    std::cout << "Warmup done. Running benchmarks (" << numIts << " iterations)..." << std::endl;
    preciseStopwatch stopwatch;
    for (size_t i = 0; i < numIts; ++i) {
        const auto objects = yoloV8.detectObjects(img);
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

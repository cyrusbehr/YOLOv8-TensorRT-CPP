#include "yolov8.h"

// Runs object detection on video stream then displays annotated results.

int main(int argc, char *argv[]) {
    // Parse the command line arguments
    // Must pass the model path and image path as a command line argument to the executable
    if (argc < 3) {
        std::cout << "Error: Must specify the model and input image" << std::endl;
        std::cout << "Usage: " << argv[0] << "/path/to/onnx/model.onnx /path/to/your/image.jpg" << std::endl;
        return -1;
    }

    if (argc > 4) {
        std::cout << "Error: Too many arguments provided" << std::endl;
        std::cout << "Usage: " << argv[0] << "/path/to/onnx/model.onnx /path/to/your/image.jpg" << std::endl;
    }

    // Ensure the onnx model exists
    const std::string onnxModelPath = argv[1];
    if (!doesFileExist(onnxModelPath)) {
        std::cout << "Error: Unable to find file at path: " << onnxModelPath << std::endl;
        return -1;
    }

    // Create our YoloV8 engine
    // Use default probability threshold, nms threshold, and top k
    YoloV8 yoloV8(onnxModelPath);

    // Initialize the video stream
    cv::VideoCapture cap;
    // TODO: Replace this with your video source.
    // 0 is default webcam, but can replace with string such as "/dev/video0", or an RTSP stream URL
    cap.open(0);

    if (!cap.isOpened()) {
        throw std::runtime_error("Unable to open video capture!");
    }

    while (true) {
        // Grab frame
        cv::Mat img;
        cap >> img;

        if (img.empty()) {
            throw std::runtime_error("Unable to decode image from video stream.");
        }

        // Run inference
        const auto objects = yoloV8.detectObjects(img);

        // Draw the bounding boxes on the image
        yoloV8.drawObjectLabels(img, objects);

        // Display the results
        cv::imshow("Object Detection", img);
        if (cv::waitKey(1) >= 0) {
            break;
        }
    }
    return 0;
}

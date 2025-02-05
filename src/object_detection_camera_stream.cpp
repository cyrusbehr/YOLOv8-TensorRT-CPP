#include "cmd_line_util.h"
#include "yolov8.h"
#include <opencv2/cudaimgproc.hpp>

// Runs object detection on video stream then displays annotated results.
int main(int argc, char *argv[]) {
    YoloV8Config config;
    std::string onnxModelPath;
    std::string trtModelPath;
    std::string inputVideo;

    // Parse the command line arguments
    if (!parseArgumentsVideo(argc, argv, config, onnxModelPath, trtModelPath, inputVideo)) {
        return -1;
    }

    // Create the YoloV8 engine
    YoloV8 yoloV8(onnxModelPath, trtModelPath, config);

    // Define GStreamer pipeline for the CSI camera
    std::string gst_pipeline = 
        "nvarguscamerasrc ! video/x-raw(memory:NVMM), "
        "width=1280, height=720, framerate=30/1 ! nvvidconv ! "
        "video/x-raw, format=(string)BGRx ! videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=true sync=false";

    // Open the CSI camera
    cv::VideoCapture cap(gst_pipeline, cv::CAP_GSTREAMER);

    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open CSI camera" << std::endl;
        return -1;
    }

    std::cout << "CSI Camera opened successfully!" << std::endl;

    while (true) {
        // Grab frame
        cv::Mat img;
        cap >> img;

        if (img.empty())
            throw std::runtime_error("Unable to decode image from video stream.");

        // Run inference
        const auto objects = yoloV8.detectObjects(img);

        // Draw the bounding boxes on the image
        yoloV8.drawObjectLabels(img, objects);

        // Display the results
        cv::imshow("Object Detection", img);
        if (cv::waitKey(1) >= 0)
            break;
    }

    return 0;
}
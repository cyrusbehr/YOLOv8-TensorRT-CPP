#include "yolov8.h"
#include "cmd_line_util.h"
#include <opencv2/cudaimgproc.hpp>


//
static std::string create_capture(int width, int height, int fps)
{
    std::stringstream pipeline_str;
    pipeline_str << "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)"
        << std::to_string(width) << ", height=(int)" << std::to_string(height)
        << ", format=(string)NV12, framerate=(fraction)" << std::to_string(fps)
        << "/1 ! nvvidconv ! video/x-raw, format=(string)I420 ! videoconvert"
        " ! video/x-raw, format=(string)BGR ! appsink ";

    return pipeline_str.str();
}

// Runs object detection on video stream then displays annotated results.
int main(int argc, char* argv[]) {
    YoloV8Config config;
    std::string onnxModelPath;
    std::string inputVideo;

	// Parse the command line arguments
	if (!parseArgumentsVideo(argc, argv, config, onnxModelPath, inputVideo)) {
		return -1;
    }

	// Create the YoloV8 engine
	YoloV8 yoloV8(onnxModelPath, config);

	// Initialize the video stream
	cv::VideoCapture cap;
	int fps = 30;

	// Open video capture
	try {
		cap.open(create_capture(1280, 720, fps), cv::CAP_GSTREAMER);
	} catch (const std::exception& e) {
		cap.open(inputVideo);
	}

	// Try to use HD resolution (or closest resolution)
	auto resW = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	auto resH = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	std::cout << "Video resolution: (" << resW << "x" << resH << ")" << std::endl;

	if (!cap.isOpened())
		throw std::runtime_error("Unable to open video capture with input '" + inputVideo + "'");

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
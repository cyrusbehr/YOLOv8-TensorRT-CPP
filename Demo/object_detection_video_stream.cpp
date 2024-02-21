#include "yolov8.h"
#include "cmd_line_util.h"
#include <opencv2/cudaimgproc.hpp>


// Runs object detection on video stream then displays annotated results.
int main(int argc, char* argv[]) {
	EngineOptions engineOptions;
    YoloV8Config yoloV8Config;
    std::string onnxModelPath;
    std::string inputVideo;

	engineOptions.maxBatchSize = 1;
	yoloV8Config.classNames = {
		"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
		"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
		"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
		"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
		"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
		"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
		"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
		"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
		"hair drier", "toothbrush"
	};

	// Parse the command line arguments
	if (!parseArgumentsVideo(argc, argv, engineOptions, yoloV8Config, onnxModelPath, inputVideo)) {
		return -1;
    }

	// Create the YoloV8 engine
	YoloV8 yoloV8;
	if (!yoloV8.loadEngine(onnxModelPath, engineOptions, yoloV8Config))
	{
		std::cout << "Error: Unable to load onnx model at path: " << onnxModelPath << std::endl;
		return -1;
	}

	// Initialize the video stream
	cv::VideoCapture cap;

	// Open video capture
	try {
		cap.open(std::stoi(inputVideo));
	} catch (const std::exception& e) {
		cap.open(inputVideo);
	}

	// Try to use HD resolution (or closest resolution)
	auto resW = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	auto resH = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	std::cout << "Original video resolution: (" << resW << "x" << resH << ")" << std::endl;
	cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
	resW = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	resH = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	std::cout << "New video resolution: (" << resW << "x" << resH << ")" << std::endl;

	if (!cap.isOpened())
		throw std::runtime_error("Unable to open video capture with input '" + inputVideo + "'");

	while (true) {
		// Grab frame
		cv::Mat img;
		cap >> img;

		if (img.empty())
			throw std::runtime_error("Unable to decode image from video stream.");

		// Run inference
		std::vector<InferenceObject> inferenceObjects;
		if (!yoloV8.infer(img, inferenceObjects))
		{
			std::cout << "Inference failure.";
			return -1;
		}

		// Draw the bounding boxes on the image
		yoloV8.drawObjectLabels(img, inferenceObjects);

		// Display the results
		cv::imshow("Object Detection", img);
		if (cv::waitKey(1) >= 0)
			break;
	}

	return 0;
}
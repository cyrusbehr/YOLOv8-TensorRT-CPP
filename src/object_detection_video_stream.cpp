#include "yolov8.h"
#include <opencv2/cudaimgproc.hpp>

YoloV8Config config;
std::string onnxModelPath;
std::string videoCaptureInput;

void showHelp(char* argv[]) {
	std::cout << "Usage: " << argv[0] << " [OPTIONS]" << std::endl << std::endl;

	std::cout << "Options:" << std::endl;
	std::cout << "--model <string>                  Path to the ONNX model. (Mandatory)" << std::endl;
	std::cout << "--input <string || int>           Input source for detection. Accepts either a path to a video file or an integer index representing a webcam. (Mandatory)" << std::endl;
	std::cout << "--prob-threshold <float>          Sets the probability threshold for object detection. Objects with confidence scores lower than this value will be ignored. (Default: 0.25)" << std::endl;
	std::cout << "--nms-threshold <float>           Sets the Non-Maximum Suppression (NMS) threshold. NMS is used to eliminate duplicate and overlapping detections. (Default: 0.65)" << std::endl;
	std::cout << "--top-k <int>                     Sets the maximum number of top-scoring objects to be displayed by the detector. (Default: 100)" << std::endl;
	std::cout << "--seg-channels <int>              Sets the number of segmentation channels used for object segmentation. (Default: 32)" << std::endl;
	std::cout << "--seg-h <int>                     Sets the height of the segmentation mask. (Default: 160)" << std::endl;
	std::cout << "--seg-w <int>                     Sets the width of the segmentation mask. (Default: 160)" << std::endl;
	std::cout << "--seg-threshold <float>           Sets the segmentation threshold for object segmentation. This value determines the sensitivity of the segmentation mask generation. (Default: 0.5)" << std::endl;
	std::cout << "--class-names <string list>       Sets the names of object classes to be recognized by the detector. Provide the class names separated by spaces. (Default: COCO class names)" << std::endl << std::endl;
	
	std::cout << "Example usage:" << std::endl;
	std::cout << argv[0] << " --model model.onnx --input 0 --prob-threshold 0.3 --nms-threshold 0.5 --top-k 50 --seg-channels 64 --seg-h 192 --seg-w 192 --seg-threshold 0.4 --class-names cat dog car person" << std::endl;
}

bool tryGetNextArgument(int argc, char* argv[], int& currentIndex, std::string& value, std::string flag, bool printErrors = true) {
	if (currentIndex + 1 >= argc) {
		if (printErrors)
			std::cout << "Error: No arguments provided for flag '" << flag << "'" << std::endl;
		return false;
	}

	std::string nextArgument = argv[currentIndex + 1];
	if (nextArgument.substr(0, 2) == "--") {
		if (printErrors)
			std::cout << "Error: No arguments provided for flag '" << flag << "'" << std::endl;
		return false;
	}

	value = argv[++currentIndex];
	return true;
}

bool tryParseInt(std::string s, int& value, std::string flag) {
	try {
		value = std::stoi(s);
		return true;
	}
	catch (const std::exception e) {
		std::cout << "Error: Could not parse '" << s << "' as an integer for flag '" << flag << "'" << std::endl;
		return false;
	}
}

bool tryParseFloat(std::string s, float& value, std::string flag) {
	try {
		value = std::stof(s);
		return true;
	}
	catch (const std::exception e) {
		std::cout << "Error: Could not parse '" << s << "' as a float for flag '" << flag << "'" << std::endl;
		return false;
	}
}

bool parseArguments(int argc, char* argv[]) {
	if (argc == 1) {
		showHelp(argv);
		return false;
	}

	for (int i = 1; i < argc; i++) {
		std::string argument = argv[i];

		if (argument.substr(0, 2) == "--") {
			std::string flag = argument.substr(2);
			std::string nextArgument;

			if (flag == "model") {
				if (!tryGetNextArgument(argc, argv, i, nextArgument, flag))
					return false;

				if (!doesFileExist(nextArgument)) {
					std::cout << "Error: Unable to find model at path '" << nextArgument << "' for flag '" << flag << "'" << std::endl;
					return false;
				}

				onnxModelPath = nextArgument;
			}

			else if (flag == "input") {
				if (!tryGetNextArgument(argc, argv, i, nextArgument, flag))
					return false;

				videoCaptureInput = nextArgument;
			}

			else if (flag == "prob-threshold") {
				if (!tryGetNextArgument(argc, argv, i, nextArgument, flag))
					return false;

				float value;
				if (!tryParseFloat(nextArgument, value, flag))
					return false;

				config.probabilityThreshold = value;
			}

			else if (flag == "nms-threshold") {
				if (!tryGetNextArgument(argc, argv, i, nextArgument, flag))
					return false;

				float value;
				if (!tryParseFloat(nextArgument, value, flag))
					return false;

				config.nmsThreshold = value;
			}

			else if (flag == "top-k") {
				if (!tryGetNextArgument(argc, argv, i, nextArgument, flag))
					return false;

				int value;
				if (!tryParseInt(nextArgument, value, flag))
					return false;

				config.topK = value;
			}

			else if (flag == "seg-channels") {
				if (!tryGetNextArgument(argc, argv, i, nextArgument, flag))
					return false;

				int value;
				if (!tryParseInt(nextArgument, value, flag))
					return false;

				config.segChannels = value;
			}

			else if (flag == "seg-h") {
				if (!tryGetNextArgument(argc, argv, i, nextArgument, flag))
					return false;

				int value;
				if (!tryParseInt(nextArgument, value, flag))
					return false;

				config.segH = value;
			}

			else if (flag == "seg-w") {
				if (!tryGetNextArgument(argc, argv, i, nextArgument, flag))
					return false;

				int value;
				if (!tryParseInt(nextArgument, value, flag))
					return false;

				config.segW = value;
			}

			else if (flag == "seg-threshold") {
				if (!tryGetNextArgument(argc, argv, i, nextArgument, flag))
					return false;

				float value;
				if (!tryParseFloat(nextArgument, value, flag))
					return false;

				config.segmentationThreshold = value;
			}

			else if (flag == "class-names") {
				std::vector<std::string> values;
				while (tryGetNextArgument(argc, argv, i, nextArgument, flag, false)) {
					values.push_back(nextArgument);
				}

				if (values.size() == 0) {
					std::cout << "Error: No arguments provided for flag '" << flag << "'" << std::endl;
					return false;
				}

				config.classNames = values;
			}

			else {
				std::cout << "Error: Unknown flag '" << flag << "'" << std::endl;
				showHelp(argv);
				return false;
			}
		}
		else {
			std::cout << "Error: Unknown argument '" << argument << "'" << std::endl;
			showHelp(argv);
			return false;
		}
	}

	if (onnxModelPath.empty()) {
		std::cout << "Error: No arguments provided for flag 'model'" << std::endl;
		return false;
	}

	if (videoCaptureInput.empty()) {
		std::cout << "Error: No arguments provided for flag 'input'" << std::endl;
		return false;
	}

	return true;
}

// Runs object detection on video stream then displays annotated results.

int main(int argc, char* argv[]) {
	// Parse the command line arguments
	if (!parseArguments(argc, argv))
		return -1;

	// Create the YoloV8 engine
	YoloV8 yoloV8(onnxModelPath, config);

	// Initialize the video stream
	cv::VideoCapture cap;

	// Open video capture
	try {
		cap.open(std::stoi(videoCaptureInput));
	}
	catch (std::exception e) {
		cap.open(videoCaptureInput);
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
		throw std::runtime_error("Unable to open video capture with input '" + videoCaptureInput + "'");

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
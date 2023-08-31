#include "yolov8.h"
#include "cmd_line_util.h"


// Runs object detection on an input image then saves the annotated image to disk.
int main(int argc, char *argv[]) {
    EngineOptions engineOptions;
    YoloV8Config yoloV8Config;
    std::string onnxModelPath;
    std::string inputImage;

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
	if (!parseArguments(argc, argv, engineOptions, yoloV8Config, onnxModelPath, inputImage)) {
		return -1;
    }

    // Create the YoloV8 engine
    YoloV8 yoloV8;
    if (!yoloV8.loadEngine(onnxModelPath, engineOptions, yoloV8Config))
    {
        std::cout << "Error: Unable to load onnx model at path: " << onnxModelPath << std::endl;
        return -1;
    }

    // Read the input image
    auto img = cv::imread(inputImage);
    if (img.empty()) {
        std::cout << "Error: Unable to read image at path '" << inputImage << "'" << std::endl;
        return -1;
    }

    // Run inference
    std::vector<InferenceObject> inferenceObjects;
    if (!yoloV8.infer(img, inferenceObjects))
    {
        std::cout << "Inference failure.";
        return -1;
    }

    // Draw the bounding boxes on the image
    yoloV8.drawObjectLabels(img, inferenceObjects);

    std::cout << "Detected " << inferenceObjects.size() << " objects" << std::endl;

    cv::imshow("result", img);
    cv::waitKey(0);
    // Save the image to disk
    // const auto outputName = inputImage.substr(0, inputImage.find_last_of('.')) + "_annotated.jpg";
    // cv::imwrite(outputName, img);
    // std::cout << "Saved annotated image to: " << outputName << std::endl;

    return 0;
}
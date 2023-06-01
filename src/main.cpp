#include "yolov8.h"

int main(int argc, char *argv[]) {
    // Parse the command line arguments
    // Must pass the path to the input image as a command line argument to the executable
    if (argc < 2) {
        std::cout << "Error: Must specify input image" << std::endl;
        std::cout << "Usage: " << argv[0] << " /path/to/your/image.jpg" << std::endl;
        return -1;
    }

    if (argc > 2) {
        std::cout << "Error: Too many arguments provided" << std::endl;
        std::cout << "Usage: " << argv[0] << " /path/to/your/image.jpg" << std::endl;
    }

    // Ensure the image exists on disk
    const std::string inputImage = argv[1];
    if (!doesFileExist(inputImage)) {
        std::cout << "Error: The input file does not exist" << std::endl;
        return -1;
    }

    // TODO: Specify the model we want to load
    const std::string onnxModelPath = "../models/yolov8m.onnx";

    // Create our YoloV8 engine
    // Use default probability threshold, nms threshold, and top k
    YoloV8 yoloV8(onnxModelPath);

    // Read the input image
    auto img = cv::imread(inputImage);
    if (img.empty()) {
        std::cout << "Error: Unable to read image at path: " << inputImage << std::endl;
        return -1;
    }

    // Run inference
    const auto objects = yoloV8.detectObjects(img);

    // Draw the bounding boxes on the image
    yoloV8.drawObjectLabels(img, objects);

    // Save the image to disk
    const auto outputName = inputImage.substr(0, inputImage.find_last_of('.')) + "_annotated.jpg";
    cv::imwrite(outputName, img);
    std::cout << "Saved annotated image to: " << outputName << std::endl;

    return 0;
}

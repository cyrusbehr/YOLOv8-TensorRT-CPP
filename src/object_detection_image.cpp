#include "yolov8.h"

// Runs inference on an input image then saves the annotated image to disk.

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

    // Ensure the image exists on disk
    const std::string inputImage = argv[2];
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

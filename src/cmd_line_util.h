#pragma once
#include <iostream>
#include "yolov8.h"

inline void showHelp(char* argv[]) {
    std::cout << "Usage: " << argv[0] << " [OPTIONS]" << std::endl << std::endl;

    std::cout << "Options:" << std::endl;
    std::cout << "--model <string>                  Path to the ONNX model. (Mandatory)" << std::endl;
    std::cout << "--input <string || int>           Input source for detection. Accepts a path to an image file. (Mandatory)" << std::endl;
    std::cout << "--prob-threshold <float>          Sets the probability threshold for object detection. Objects with confidence scores lower than this value will be ignored. (Default: 0.25)" << std::endl;
    std::cout << "--nms-threshold <float>           Sets the Non-Maximum Suppression (NMS) threshold. NMS is used to eliminate duplicate and overlapping detections. (Default: 0.65)" << std::endl;
    std::cout << "--top-k <int>                     Sets the maximum number of top-scoring objects to be displayed by the detector. (Default: 100)" << std::endl;
    std::cout << "--seg-channels <int>              Sets the number of segmentation channels used for object segmentation. (Default: 32)" << std::endl;
    std::cout << "--seg-h <int>                     Sets the height of the segmentation mask. (Default: 160)" << std::endl;
    std::cout << "--seg-w <int>                     Sets the width of the segmentation mask. (Default: 160)" << std::endl;
    std::cout << "--seg-threshold <float>           Sets the segmentation threshold for object segmentation. This value determines the sensitivity of the segmentation mask generation. (Default: 0.5)" << std::endl;
    std::cout << "--class-names <string list>       Sets the names of object classes to be recognized by the detector. Provide the class names separated by spaces. (Default: COCO class names)" << std::endl << std::endl;

    std::cout << "Example usage:" << std::endl;
    std::cout << argv[0] << " --model model.onnx --input image.png --prob-threshold 0.3 --nms-threshold 0.5 --top-k 50 --seg-channels 64 --seg-h 192 --seg-w 192 --seg-threshold 0.4 --class-names cat dog car person" << std::endl;
}

inline bool tryGetNextArgument(int argc, char* argv[], int& currentIndex, std::string& value, std::string flag, bool printErrors = true) {
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

inline bool tryParseInt(const std::string& s, int& value, const std::string& flag) {
    try {
        value = std::stoi(s);
        return true;
    }
    catch (const std::exception& e) {
        std::cout << "Error: Could not parse '" << s << "' as an integer for flag '" << flag << "'" << std::endl;
        return false;
    }
}

inline bool tryParseFloat(const std::string& s, float& value, const std::string& flag) {
    try {
        value = std::stof(s);
        return true;
    }
    catch (const std::exception& e) {
        std::cout << "Error: Could not parse '" << s << "' as a float for flag '" << flag << "'" << std::endl;
        return false;
    }
}

inline bool parseArguments(int argc, char* argv[], YoloV8Config& config, std::string& onnxModelPath, std::string& inputImage) {
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

                if (!doesFileExist(nextArgument)) {
                    std::cout << "Error: Unable to find image at path '" << nextArgument << "' for flag '" << flag << "'" << std::endl;
                    return false;
                }

                inputImage = nextArgument;
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

    if (inputImage.empty()) {
        std::cout << "Error: No arguments provided for flag 'input'" << std::endl;
        return false;
    }

    return true;
}

inline bool parseArgumentsVideo(int argc, char* argv[], YoloV8Config& config, std::string& onnxModelPath, std::string& inputVideo) {
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

                inputVideo = nextArgument;
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

    if (inputVideo.empty()) {
        std::cout << "Error: No arguments provided for flag 'input'" << std::endl;
        return false;
    }

    return true;
}
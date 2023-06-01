#pragma once
#include "engine.h"
#include <fstream>

// Utility method for checking if a file exists on disk
inline bool doesFileExist (const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

struct Object {
    // The object class.
    int label{};
    // The detection's confidence probability.
    float probability{};
    // The object bounding box rectangle.
    cv::Rect_<float> rect;
};

class YoloV8 {
public:
    // Builds the onnx model into a TensorRT engine, and loads the engine into memory
    YoloV8(const std::string& onnxModelPath, const float probabilityThreshold = 0.25f, const float nmsThreshold = 0.65f, const int topK = 100);

    // Detect the objects in the image
    std::vector<Object> detectObjects(const cv::Mat& inputImgBGR);

    // Draw the object bounding boxes and labels on the image
    void drawObjectLabels(cv::Mat& image, const std::vector<Object> &objects, unsigned int scale = 2);
private:
    // Preprocess the input
    std::vector<std::vector<cv::cuda::GpuMat>> preprocess(const cv::Mat& inputImgBGR);

    // Postpreocess the output
    std::vector<Object> postprocess(std::vector<float>& featureVector);

    // Utility method for transforming nested array into single array
    void transformOutput(std::vector<std::vector<std::vector<float>>>& input, std::vector<float>& output);

    std::unique_ptr<Engine> m_trtEngine = nullptr;

    // Used for image preprocessing
    // YoloV8 model expects values between [0.f, 1.f] so we use the following params
    const std::array<float, 3> SUB_VALS {0.f, 0.f, 0.f};
    const std::array<float, 3> DIV_VALS {1.f, 1.f, 1.f};
    const bool NORMALIZE = true;

    float m_ratio;
    float m_imgWidth;
    float m_imgHeight;

    // Filter thresholds
    const float PROBABILITY_THRESHOLD;
    const float NMS_THRESHOLD;
    const int TOP_K;

    /** Object classes as stings. */
    const std::vector<std::string> classNames = {
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

    // Color list for drawing objects
    const float COLOR_LIST[80][3] = {
            {1, 1, 1},
            {0.098, 0.325, 0.850},
            {0.125, 0.694, 0.929},
            {0.556, 0.184, 0.494},
            {0.188, 0.674, 0.466},
            {0.933, 0.745, 0.301},
            {0.184, 0.078, 0.635},
            {0.300, 0.300, 0.300},
            {0.600, 0.600, 0.600},
            {0.000, 0.000, 1.000},
            {0.000, 0.500, 1.000},
            {0.000, 0.749, 0.749},
            {0.000, 1.000, 0.000},
            {1.000, 0.000, 0.000},
            {1.000, 0.000, 0.667},
            {0.000, 0.333, 0.333},
            {0.000, 0.667, 0.333},
            {0.000, 1.000, 0.333},
            {0.000, 0.333, 0.667},
            {0.000, 0.667, 0.667},
            {0.000, 1.000, 0.667},
            {0.000, 0.333, 1.000},
            {0.000, 0.667, 1.000},
            {0.000, 1.000, 1.000},
            {0.500, 0.333, 0.000},
            {0.500, 0.667, 0.000},
            {0.500, 1.000, 0.000},
            {0.500, 0.000, 0.333},
            {0.500, 0.333, 0.333},
            {0.500, 0.667, 0.333},
            {0.500, 1.000, 0.333},
            {0.500, 0.000, 0.667},
            {0.500, 0.333, 0.667},
            {0.500, 0.667, 0.667},
            {0.500, 1.000, 0.667},
            {0.500, 0.000, 1.000},
            {0.500, 0.333, 1.000},
            {0.500, 0.667, 1.000},
            {0.500, 1.000, 1.000},
            {1.000, 0.333, 0.000},
            {1.000, 0.667, 0.000},
            {1.000, 1.000, 0.000},
            {1.000, 0.000, 0.333},
            {1.000, 0.333, 0.333},
            {1.000, 0.667, 0.333},
            {1.000, 1.000, 0.333},
            {1.000, 0.000, 0.667},
            {1.000, 0.333, 0.667},
            {1.000, 0.667, 0.667},
            {1.000, 1.000, 0.667},
            {1.000, 0.000, 1.000},
            {1.000, 0.333, 1.000},
            {1.000, 0.667, 1.000},
            {0.000, 0.000, 0.333},
            {0.000, 0.000, 0.500},
            {0.000, 0.000, 0.667},
            {0.000, 0.000, 0.833},
            {0.000, 0.000, 1.000},
            {0.000, 0.167, 0.000},
            {0.000, 0.333, 0.000},
            {0.000, 0.500, 0.000},
            {0.000, 0.667, 0.000},
            {0.000, 0.833, 0.000},
            {0.000, 1.000, 0.000},
            {0.167, 0.000, 0.000},
            {0.333, 0.000, 0.000},
            {0.500, 0.000, 0.000},
            {0.667, 0.000, 0.000},
            {0.833, 0.000, 0.000},
            {1.000, 0.000, 0.000},
            {0.000, 0.000, 0.000},
            {0.143, 0.143, 0.143},
            {0.286, 0.286, 0.286},
            {0.429, 0.429, 0.429},
            {0.571, 0.571, 0.571},
            {0.714, 0.714, 0.714},
            {0.857, 0.857, 0.857},
            {0.741, 0.447, 0.000},
            {0.741, 0.717, 0.314},
            {0.000, 0.500, 0.500}
    };
};
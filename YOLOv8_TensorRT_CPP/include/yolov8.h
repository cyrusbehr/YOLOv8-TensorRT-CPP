#pragma once
#include <engine.h>

#ifdef _MSC_VER
#define YOLOV8_HIDDEN_API
#ifdef YOLOV8_TENSORRT_CPP_EXPORT
#define YOLOV8_API __declspec(dllexport)
#else
#define YOLOV8_API __declspec(dllimport)
#endif // YOLOV8_TENSORRT_CPP_EXPORT
#else
#define YOLOV8_API __attribute((visibility("default")))
#define YOLOV8_HIDDEN_API __attribute((visibility("hidden")))
#endif // _MSV_VER

struct InferenceObject {
    // The object class.
    int label{};
    // The detection's confidence probability.
    float probability{};
    // The object bounding box rectangle.
    cv::Rect_<float> rect;
    // Semantic segmentation mask
    cv::Mat boxMask;
    // Pose estimation key points
    std::vector<float> kps{};
};

// Config the behavior of the YoloV8 detector.
// Can pass these arguments as command line parameters.
struct YoloV8Config {
    // Probability threshold used to filter detected objects
    float probabilityThreshold = 0.25f;
    // Non-maximum suppression threshold
    float nmsThreshold = 0.65f;
    // Max number of detected objects to return
    int topK = 100;
    // Segmentation config options
    int segChannels = 32;
    int segH = 160;
    int segW = 160;
    float segmentationThreshold = 0.5f;
    // Pose estimation options
    int numKPS = 17;
    float kpsThreshold = 0.5f;
    // Class thresholds
    std::vector<std::string> classNames = {};
};

class YoloV8 {
public:
    YOLOV8_API YoloV8();
    YOLOV8_API ~YoloV8();

    YOLOV8_API bool loadEngine(const std::string& onnxPath,
                               const EngineOptions& engineOptions,
                               const YoloV8Config& yoloV8Config,
                               bool autoBuild = true);

    YOLOV8_API bool infer(const std::string& imagePath, std::vector<InferenceObject>& inferenceObjects);
    YOLOV8_API bool infer(const cv::Mat& inputImage, std::vector<InferenceObject>& inferenceObjects);
    YOLOV8_API bool infer(const cv::cuda::GpuMat& inputImage, std::vector<InferenceObject>& inferenceObjects);

    // Draw the object bounding boxes and labels on the image
    YOLOV8_API void drawObjectLabels(cv::Mat& image, const std::vector<InferenceObject> &objects, unsigned int scale = 2);
private:
    // Preprocess the input
    void preprocess(const cv::cuda::GpuMat& gpuImg, std::vector<std::vector<cv::cuda::GpuMat>>& inputs);

    // Postprocess the output
    void postprocessDetection(std::vector<float>& featureVector, std::vector<InferenceObject>& inferenceObjects);

    // Postprocess the output for pose model
    void postprocessPose(std::vector<float>& featureVector, std::vector<InferenceObject>& inferenceObjects);

    // Postprocess the output for segmentation model
    void postprocessSegmentation(std::vector<std::vector<float>>& featureVectors,
                                 std::vector<InferenceObject>& inferenceObjects);

    std::unique_ptr<Engine> m_trtEngine;

    // Used for image preprocessing
    // YoloV8 model expects values between [0.f, 1.f] so we use the following params
    const std::array<float, 3> SUB_VALS {0.f, 0.f, 0.f};
    const std::array<float, 3> DIV_VALS {1.f, 1.f, 1.f};
    const bool NORMALIZE = true;

    float m_ratio = 1;
    float m_imgWidth = 0;
    float m_imgHeight = 0;

    // Filter thresholds
    float m_probabilityThreshold;
    float m_nmsThreshold;
    int m_topK;

    // Segmentation constants
    int m_segChannels;
    int m_segH;
    int m_segW;
    float m_segmentationThreshold;

    // Pose estimation constant
    int m_numKps;
    float m_kpsThreshold;

    // InferenceObject classes as strings
    std::vector<std::string> m_classNames;

    // Color list for drawing objects
    const std::vector<cv::Scalar> COLOR_LIST = {
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

    const std::vector<std::vector<unsigned int>> KPS_COLORS = {
            {0, 255, 0},
            {0, 255, 0},
            {0, 255, 0},
            {0, 255, 0},
            {0, 255, 0},
            {255, 128, 0},
            {255, 128, 0},
            {255, 128, 0},
            {255, 128, 0},
            {255, 128, 0},
            {255, 128, 0},
            {51, 153, 255},
            {51, 153, 255},
            {51, 153, 255},
            {51, 153, 255},
            {51, 153, 255},
            {51, 153, 255}
    };

    const std::vector<std::vector<unsigned int>> SKELETON = {
            {16, 14},
            {14, 12},
            {17, 15},
            {15, 13},
            {12, 13},
            {6, 12},
            {7, 13},
            {6, 7},
            {6, 8},
            {7, 9},
            {8, 10},
            {9, 11},
            {2, 3},
            {1, 2},
            {1, 3},
            {2, 4},
            {3, 5},
            {4, 6},
            {5, 7}
    };

    const std::vector<std::vector<unsigned int>> LIMB_COLORS = {
            {51, 153, 255},
            {51, 153, 255},
            {51, 153, 255},
            {51, 153, 255},
            {255, 51, 255},
            {255, 51, 255},
            {255, 51, 255},
            {255, 128, 0},
            {255, 128, 0},
            {255, 128, 0},
            {255, 128, 0},
            {255, 128, 0},
            {0, 255, 0},
            {0, 255, 0},
            {0, 255, 0},
            {0, 255, 0},
            {0, 255, 0},
            {0, 255, 0},
            {0, 255, 0}
    };
};
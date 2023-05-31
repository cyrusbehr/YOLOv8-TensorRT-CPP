#include <opencv2/opencv.hpp>
#include <chrono>
#include <fstream>
#include "engine.h"

typedef std::chrono::high_resolution_clock Clock;

/** Object classes. */
enum class ObjectLabel {person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic_light,
    fire_hydrant, stop_sign, parking_meter, bench, bird, cat, dog, horse, sheep, cow,
    elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee,
    skis, snowboard, sports_ball, kite, baseball_bat, baseball_glove, skateboard, surfboard,
    tennis_racket, bottle, wine_glass, cup, fork, knife, spoon, bowl, banana, apple,
    sandwich, orange, broccoli, carrot, hot_dog, pizza, donut, cake, chair, couch,
    potted_plant, bed, dining_table, toilet, tv, laptop, mouse, remote, keyboard, cell_phone,
    microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy_bear,
    hair_drier, toothbrush, COUNT};


struct BoundingBox {
    /** The object class as a label. */
    ObjectLabel label;
    /** The detection's confidence probability. */
    float probability;
    /** The top left corner Point of the bounding box. */
    cv::Point topLeft;
    /** The bottom right corner Point of the bounding box. */
    cv::Point bottomRight;
};

inline bool doesFileExist (const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

// Decode the raw output for generate the yolo proposals
void generateProposals(const std::vector<float>& featureBlob, float probThreshold, std::vector<cv::Rect>& proposals,
                                                std::vector<float>& confVec, std::vector<int>& classVec) {
    auto numClasses = static_cast<int>(ObjectLabel::COUNT);
    auto dets = featureBlob.size() / (numClasses + 5);
    for (unsigned int boxIdx = 0; boxIdx < dets; boxIdx++) {
        const int basicPos = boxIdx * (numClasses + 5);
        float xCenter = featureBlob[basicPos + 0];
        float yCenter = featureBlob[basicPos + 1];
        float w = featureBlob[basicPos + 2];
        float h = featureBlob[basicPos + 3];
        float x0 = xCenter - w * 0.5f;
        float y0 = yCenter - h * 0.5f;
        float boxObjectness = featureBlob[basicPos + 4];
        for (int classIdx = 0; classIdx < numClasses; classIdx++) {
            float boxC = featureBlob[basicPos + 5 + classIdx];
            float boxProb = boxObjectness * boxC;
            if (boxProb > probThreshold) {
                cv::Rect rect;
                rect.x = x0;
                rect.y = y0;
                rect.width = w;
                rect.height = h;
                proposals.push_back(rect);
                confVec.push_back(boxProb);
                classVec.push_back(classIdx);
            }
        }
    }
}

void postProcess(const std::vector<float>& featureBlob, std::vector<BoundingBox>& objects, float scale, const int originalWidth, const int originalHeight,
                 const float probThreshold, const float nmsThreshold) {

    // Generate the yolo proposals
    std::vector<cv::Rect> proposals;
    std::vector<float> confVec;
    std::vector<int> classVec;
    generateProposals(featureBlob, probThreshold, proposals, confVec, classVec);

    // Run NMS to remove overlapping bounding boxes
    std::vector<int> indices;
    cv::dnn::NMSBoxes(proposals, confVec, probThreshold, nmsThreshold, indices);

    for (int chosenIdx : indices) {
        BoundingBox bbox{};
        bbox.probability = confVec[chosenIdx];
        bbox.label = static_cast<ObjectLabel>(classVec[chosenIdx]);

        const auto& rect = proposals[chosenIdx];

        // adjust offset to original unpadded image dimensions
        float x0 = (rect.x) / scale;
        float y0 = (rect.y) / scale;
        float x1 = (rect.x + rect.width) / scale;
        float y1 = (rect.y + rect.height) / scale;

        // Clip to ensure bounding box does not exceed the image dimensions
        bbox.topLeft.x = std::max(std::min(x0, (float)(originalWidth - 1)), 0.f);
        bbox.topLeft.y = std::max(std::min(y0, (float)(originalHeight - 1)), 0.f);
        bbox.bottomRight.x = std::max(std::min(x1, (float)(originalWidth - 1)), 0.f);
        bbox.bottomRight.y = std::max(std::min(y1, (float)(originalHeight - 1)), 0.f);
        objects.emplace_back(std::move(bbox));
    }
}

// Utility method for tranforming nested array into single array
void transformOutput(std::vector<std::vector<std::vector<float>>>& input, std::vector<float>& output) {
    if (input.size() != 1 || input[0].size() != 1) {
        throw std::logic_error("The feature vector has incorrect dimensions!");
    }

    output = std::move(input[0][0]);
}

int main(int argc, char *argv[]) {
    // Change these thresholds as needed
    const float NMS_THRESH = 0.5f;
    const float BBOX_CONF_THRESH = 0.3f;

    if (argc < 2) {
        std::cout << "Error: Must specify input image" << std::endl;
        std::cout << "Usage: " << argv[0] << " /path/to/your/image.jpg" << std::endl;
        return -1;
    }

    if (argc > 2) {
        std::cout << "Error: Too many arguments provided" << std::endl;
        std::cout << "Usage: " << argv[0] << " /path/to/your/image.jpg" << std::endl;
    }

    const std::string inputImage = argv[1];
    if (!doesFileExist(inputImage)) {
        std::cout << "Error: The input file does not exist" << std::endl;
        return -1;
    }

    // Specify options for GPU inference
    Options options;
    // This particular model only supports a fixed batch size of 1
    options.doesSupportDynamicBatchSize = false;
    options.optBatchSize = 1;
    options.maxBatchSize = 1;
    options.maxWorkspaceSize = 2000000000;

    // Use FP16 precision to speed up inference
    options.precision = Precision::FP16;

    Engine engine(options);

    // Specify the model we want to load
    const std::string onnxModelPath = "../models/yolov8m.onnx";

    // Build the onnx model into a TensorRT engine file
    // If the engine file already exists, this function will return immediately
    // The engine file is rebuilt any time the above Options are changed. 
    auto succ = engine.build(onnxModelPath);
    if (!succ) {
        std::cout << "Error: Unable to build the TensorRT engine" << std::endl;
        std::cout << "Try increasing TensorRT log severity to kVERBOSE (in /libs/tensorrt-cpp-api/engine.cpp)" << std::endl;
        return -1;
    }

    // Load the TensorRT engine file
    succ = engine.loadNetwork();
    if (!succ) {
        std::cout << "Error: Unable to load TensorRT engine weights" << std::endl;
        return -1;
    }

    // Read the input image
    auto cpuImg = cv::imread(inputImage);
    if (cpuImg.empty()) {
        std::cout << "Error: Unable to read image at path: " << inputImage << std::endl;
        return -1;
    }

    // The model expects RGB input
    cv::cvtColor(cpuImg, cpuImg, cv::COLOR_BGR2RGB);

    // Upload to GPU memory
    cv::cuda::GpuMat img;
    img.upload(cpuImg);

    // Populate the input vectors
    const auto& inputDims = engine.getInputDims();

    // Resize to the model expected input size while maintaining the aspect ratio with the use of padding
    auto resized = Engine::resizeKeepAspectRatioPadRightBottom(img, inputDims[0].d[1], inputDims[0].d[2]);
    std::vector<cv::cuda::GpuMat> input{std::move(resized)};
    std::vector<std::vector<cv::cuda::GpuMat>> inputs {std::move(input)};

    // Compute the scale factor for when we need to convert the detected bounding boxes back to the original image coordinate system
    float scale = std::min(inputDims[0].d[2] / img.cols, inputDims[0].d[1] / img.rows);

    // Define our preprocessing code
    // YoloV8 model expects values between [0.f, 1.f] so we use the following params
    std::array<float, 3> subVals {0.f, 0.f, 0.f};
    std::array<float, 3> divVals {1.f, 1.f, 1.f};
    bool normalize = true;

    // Run inference 10 times to warm up the engine
    std::cout << "Warming up engine..." << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::vector<std::vector<std::vector<float>>> featureVectors;
        succ = engine.runInference(inputs, featureVectors, subVals, divVals, normalize);
        if (!succ) {
            std::cout << "Error: Unable to run inference" << std::endl;
            return -1;
        }
    }
    std::cout << "Warmup complete" << std::endl;

    size_t numIts = 100;
    std::cout << "Benchmarking " << numIts << " inference samples..." << std::endl;
    // Now run the benchmark
    auto t1 = Clock::now();
    for (size_t i = 0; i < numIts; ++i) {
        std::vector<std::vector<std::vector<float>>> featureVectors;
        succ = engine.runInference(inputs, featureVectors, subVals, divVals, normalize);
        if (!succ) {
            std::cout << "Error: Unable to run inference" << std::endl;
            return -1;
        }

        // Include post processing in benchmark
        std::vector<float> featureVector;
        transformOutput(featureVectors, featureVector);
        std::vector<BoundingBox> bboxes;
        postProcess(featureVector, bboxes, scale, img.cols, img.rows, BBOX_CONF_THRESH, NMS_THRESH);
    }
    auto t2 = Clock::now();
    double totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    std::cout << "Average time per inference: " << totalTime / numIts / static_cast<float>(inputs[0].size()) <<
              " ms, for batch size of: " << inputs[0].size() << std::endl;



    return 0;
}

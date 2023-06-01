#include <chrono>
#include <opencv2/cudaimgproc.hpp>

#include "yolov8.h"

typedef std::chrono::high_resolution_clock Clock;

YoloV8::YoloV8(const std::string &onnxModelPath, const float probabilityThreshold, const float nmsThreshold, const int topK)
        : PROBABILITY_THRESHOLD(probabilityThreshold)
        , NMS_THRESHOLD(nmsThreshold)
        , TOP_K(topK) {
    // Specify options for GPU inference
    Options options;
    // This particular model only supports a fixed batch size of 1
    options.doesSupportDynamicBatchSize = false;
    options.optBatchSize = 1;
    options.maxBatchSize = 1;
    options.maxWorkspaceSize = 2000000000;

    // Use FP16 precision to speed up inference
    options.precision = Precision::FP16;

    // Create our TensorRT inference engine
    m_trtEngine = std::make_unique<Engine>(options);

    // Build the onnx model into a TensorRT engine file
    // If the engine file already exists, this function will return immediately
    // The engine file is rebuilt any time the above Options are changed.
    auto succ = m_trtEngine->build(onnxModelPath);
    if (!succ) {
        const std::string errMsg = "Error: Unable to build the TensorRT engine. "
                                   "Try increasing TensorRT log severity to kVERBOSE (in /libs/tensorrt-cpp-api/engine.cpp).";
        throw std::runtime_error(errMsg);
    }

    // Load the TensorRT engine file
    succ = m_trtEngine->loadNetwork();
    if (!succ) {
        throw std::runtime_error("Error: Unable to load TensorRT engine weights into memory.");
    }
}

std::vector<std::vector<cv::cuda::GpuMat>> YoloV8::preprocess(const cv::Mat &inputImgBGR) {
    // Upload the image to GPU memory
    cv::cuda::GpuMat gpuImg;
    gpuImg.upload(inputImgBGR);

    // The model expects RGB input
    cv::cuda::cvtColor(gpuImg, gpuImg, cv::COLOR_BGR2RGB);

    // Populate the input vectors
    const auto& inputDims = m_trtEngine->getInputDims();

    // Resize to the model expected input size while maintaining the aspect ratio with the use of padding
    auto resized = Engine::resizeKeepAspectRatioPadRightBottom(gpuImg, inputDims[0].d[1], inputDims[0].d[2]);

    // Convert to format expected by our inference engine
    // The reason for the strange format is because it supports models with multiple inputs as well as batching
    // In our case though, the model only has a single input and we are using a batch size of 1.
    std::vector<cv::cuda::GpuMat> input{std::move(resized)};
    std::vector<std::vector<cv::cuda::GpuMat>> inputs {std::move(input)};

    // These params will be used in the post-processing stage
    m_imgHeight = gpuImg.rows;
    m_imgHeight = gpuImg.cols;
    m_ratio =  1.f / std::min(inputDims[0].d[2] / static_cast<float>(gpuImg.cols), inputDims[0].d[1] / static_cast<float>(gpuImg.rows));

    return inputs;
}

std::vector<Object> YoloV8::detectObjects(const cv::Mat &inputImgBGR) {
    // Preprocess the input image
    const auto input = preprocess(inputImgBGR);

    // Run inference using the TensorRT engine
    std::vector<std::vector<std::vector<float>>> featureVectors;
    auto succ = m_trtEngine->runInference(input, featureVectors, SUB_VALS, DIV_VALS, NORMALIZE);
    if (!succ) {
        throw std::runtime_error("Error: Unable to run inference.");
    }

    // Since we have a batch size of 1 and only 1 output, we must convert the output from a 3D array to a 1D array.
    std::vector<float> featureVector;
    transformOutput(featureVectors, featureVector);

    // Postprocess the output
    return postprocess(featureVector);
}

void YoloV8::transformOutput(std::vector<std::vector<std::vector<float>>>& input, std::vector<float>& output) {
    if (input.size() != 1 || input[0].size() != 1) {
        throw std::logic_error("The feature vector has incorrect dimensions!");
    }

    output = std::move(input[0][0]);
}

std::vector<Object> YoloV8::postprocess(std::vector<float> &featureVector) {
    const auto& outputDims = m_trtEngine->getOutputDims();
    auto numChannels = outputDims[0].d[1];
    auto numAnchors = outputDims[0].d[2];

    auto numClasses = classNames.size();

    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<int> indices;

    cv::Mat output = cv::Mat(numChannels, numAnchors, CV_32F, featureVector.data());
    output = output.t();

    // Get all the YOLO proposals
    for (int i = 0; i < numAnchors; i++) {
        auto rowPtr = output.row(i).ptr<float>();
        auto bboxesPtr = rowPtr;
        auto scoresPtr = rowPtr + 4;
        auto maxSPtr = std::max_element(scoresPtr, scoresPtr + numClasses);
        float score = *maxSPtr;
        if (score > PROBABILITY_THRESHOLD) {
            float x = *bboxesPtr++;
            float y = *bboxesPtr++;
            float w = *bboxesPtr++;
            float h = *bboxesPtr;

            float x0 = std::clamp((x - 0.5f * w) * m_ratio, 0.f, m_imgWidth);
            float y0 = std::clamp((y - 0.5f * h) * m_ratio, 0.f, m_imgHeight);
            float x1 = std::clamp((x + 0.5f * w) * m_ratio, 0.f, m_imgWidth);
            float y1 = std::clamp((y + 0.5f * h) * m_ratio, 0.f, m_imgHeight);

            int label = maxSPtr - scoresPtr;
            cv::Rect_<float> bbox;
            bbox.x = x0;
            bbox.y = y0;
            bbox.width = x1 - x0;
            bbox.height = y1 - y0;

            bboxes.push_back(bbox);
            labels.push_back(label);
            scores.push_back(score);
        }
    }

    // Run NMS
    cv::dnn::NMSBoxes(bboxes, scores, PROBABILITY_THRESHOLD, NMS_THRESHOLD, indices);

    std::vector<Object> objects;

    // Choose the top k detections
    int cnt = 0;
    for (auto& chosenIdx : indices) {
        if (cnt >= TOP_K) {
            break;
        }

        Object obj{};
        obj.probability = scores[chosenIdx];
        obj.label = labels[chosenIdx];
        obj.rect = bboxes[chosenIdx];
        objects.push_back(obj);

        cnt += 1;
    }

    return objects;
}

void YoloV8::drawObjectLabels(cv::Mat& image, const std::vector<Object> &objects, unsigned int scale) {
    for (auto & object : objects) {
        cv::Scalar color = cv::Scalar(COLOR_LIST[object.label][0], COLOR_LIST[object.label][1],
                                      COLOR_LIST[object.label][2]);
        float meanColor = cv::mean(color)[0];
        cv::Scalar txtColor;
        if (meanColor > 0.5){
            txtColor = cv::Scalar(0, 0, 0);
        }else{
            txtColor = cv::Scalar(255, 255, 255);
        }

        const auto& rect = object.rect;

        char text[256];
        sprintf(text, "%s %.1f%%", classNames[object.label].c_str(), object.probability * 100);

        int baseLine = 0;
        cv::Size labelSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.35 * scale, scale, &baseLine);

        cv::Scalar txt_bk_color = color * 0.7 * 255;

        int x = object.rect.x;
        int y = object.rect.y + 1;

        cv::rectangle(image, rect, color * 255, scale + 1);

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(labelSize.width, labelSize.height + baseLine)),
                      txt_bk_color, -1);

        cv::putText(image, text, cv::Point(x, y + labelSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.35 * scale, txtColor, scale);
    }
}
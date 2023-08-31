#include <opencv2/cudaimgproc.hpp>
#include "yolov8.h"

#if HAS_CPP_17 == 0
float clamp(float val, float minVal, float maxVal)
{
    if (val <= minVal) return minVal;
    else if (val >= maxVal) return maxVal;

    return val;
}
#endif

YoloV8::YoloV8()
    : m_trtEngine(std::make_unique<Engine>())
    , m_probabilityThreshold(0.25f)
    , m_nmsThreshold(0.65f)
    , m_topK(100)
    , m_segChannels(32)
    , m_segH(160)
    , m_segW(160)
    , m_segmentationThreshold(0.5f)
    , m_numKps(17)
    , m_kpsThreshold(0.5f)
{
    
}

YoloV8::~YoloV8()
{
}

bool YoloV8::loadEngine(const std::string& onnxPath, const EngineOptions& engineOptions,
                        const YoloV8Config& yoloV8Config, bool autoBuild)
{
    if (autoBuild && !Engine::build(onnxPath, engineOptions))
    {
        return false;
    }

    const std::string enginePath = EngineUtil::serializeEngineOptions(engineOptions, onnxPath);

    const bool success = m_trtEngine->loadNetwork(enginePath, engineOptions, SUB_VALS, DIV_VALS, NORMALIZE);
    if (!success)
    {
        return false;
    }

    m_probabilityThreshold = yoloV8Config.probabilityThreshold;
    m_nmsThreshold = yoloV8Config.nmsThreshold;
    m_topK = yoloV8Config.topK;
    m_segChannels = yoloV8Config.segChannels;
    m_segH = yoloV8Config.segH;
    m_segW = yoloV8Config.segW;
    m_segmentationThreshold = yoloV8Config.segmentationThreshold;
    m_numKps = yoloV8Config.numKPS;
    m_kpsThreshold = yoloV8Config.kpsThreshold;
    m_classNames = yoloV8Config.classNames;

    return true;
}

bool YoloV8::infer(const std::string& imagePath, std::vector<InferenceObject>& inferenceObjects)
{
    const cv::Mat inputImage = cv::imread(imagePath);

    return infer(inputImage, inferenceObjects);
}

bool YoloV8::infer(const cv::Mat& inputImage, std::vector<InferenceObject>& inferenceObjects)
{
    if (inputImage.empty())
    {
        return false;
    }

    cv::cuda::GpuMat gpuImage;
    gpuImage.upload(inputImage);

    return infer(gpuImage, inferenceObjects);
}

bool YoloV8::infer(const cv::cuda::GpuMat& inputImage, std::vector<InferenceObject>& inferenceObjects)
{
    if (inputImage.empty())
    {
        return false;
    }

    // Preprocess the input image
#ifdef ENABLE_BENCHMARKS
    static int numIts = 1;
    preciseStopwatch s1;
#endif
    std::vector<std::vector<cv::cuda::GpuMat>> input;
    preprocess(inputImage, input);
#ifdef ENABLE_BENCHMARKS
    static long long t1 = 0;
    t1 += s1.elapsedTime<long long, std::chrono::microseconds>();
    std::cout << "Avg Preprocess time: " << (t1 / numIts) / 1000.f << " ms" << std::endl;
#endif
    // Run inference using the TensorRT engine
#ifdef ENABLE_BENCHMARKS
    preciseStopwatch s2;
#endif
    std::vector<std::vector<std::vector<float>>> featureVectors;
    auto succ = m_trtEngine->runInference(input, featureVectors);
    if (!succ) {
        throw std::runtime_error("Error: Unable to run inference.");
    }
#ifdef ENABLE_BENCHMARKS
    static long long t2 = 0;
    t2 += s2.elapsedTime<long long, std::chrono::microseconds>();
    std::cout << "Avg Inference time: " << (t2 / numIts) / 1000.f << " ms" << std::endl;
    preciseStopwatch s3;
#endif
    // Check if our model does only object detection or also supports segmentation
    const auto& numOutputs = m_trtEngine->getOutputDims().size();
    if (numOutputs == 1) {
        // InferenceObject detection or pose estimation
        // Since we have a batch size of 1 and only 1 output, we must convert the output from a 3D array to a 1D array.
        std::vector<float> featureVector;
        Engine::transformOutput(featureVectors, featureVector);

        const auto& outputDims = m_trtEngine->getOutputDims();
        const int numChannels = outputDims[outputDims.size() - 1].d[1];
        // TODO: Need to improve this to make it more generic (don't use magic number).
        // For now it works with Ultralytics pretrained models.
        if (numChannels == 56) {
            // Pose estimation
            postprocessPose(featureVector, inferenceObjects);
        }
        else {
            // InferenceObject detection
            postprocessDetection(featureVector, inferenceObjects);
        }
    }
    else {
        // Segmentation
        // Since we have a batch size of 1 and 2 outputs, we must convert the output from a 3D array to a 2D array.
        std::vector<std::vector<float>> featureVector;
        Engine::transformOutput(featureVectors, featureVector);
        postprocessSegmentation(featureVector, inferenceObjects);
    }
#ifdef ENABLE_BENCHMARKS
    static long long t3 = 0;
    t3 += s3.elapsedTime<long long, std::chrono::microseconds>();
    std::cout << "Avg Postprocess time: " << (t3 / numIts++) / 1000.f << " ms\n" << std::endl;
#endif
    return true;
}

void YoloV8::drawObjectLabels(cv::Mat& image, const std::vector<InferenceObject>& objects, unsigned int scale) {
    // If segmentation information is present, start with that
    if (!objects.empty() && !objects[0].boxMask.empty()) {
        cv::Mat mask = image.clone();
        for (const auto& object : objects) {
            // Choose the color
            int colorIndex = object.label % COLOR_LIST.size(); // We have only defined 80 unique colors
            const cv::Scalar& color = COLOR_LIST.at(colorIndex);

            // Add the mask for said object
            mask(object.rect).setTo(color * 255, object.boxMask);
        }
        // Add all the masks to our image
        cv::addWeighted(image, 0.5, mask, 0.8, 1, image);
    }

    // Bounding boxes and annotations
    for (auto& object : objects) {
        // Choose the color
        int colorIndex = object.label % COLOR_LIST.size(); // We have only defined 80 unique colors
        const cv::Scalar& color = COLOR_LIST.at(colorIndex);
        float meanColor = cv::mean(color)[0];
        cv::Scalar txtColor;
        if (meanColor > 0.5) {
            txtColor = cv::Scalar(0, 0, 0);
        }
        else {
            txtColor = cv::Scalar(255, 255, 255);
        }

        const auto& rect = object.rect;

        // Draw rectangles and text
        char text[256];
        sprintf(text, "%s %.1f%%", m_classNames[object.label].c_str(), object.probability * 100);

        int baseLine = 0;
        cv::Size labelSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.35 * scale, scale, &baseLine);

        cv::Scalar txt_bk_color = color * 0.7 * 255;

        int x = object.rect.x;
        int y = object.rect.y + 1;

        cv::rectangle(image, rect, color * 255, scale + 1);

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(labelSize.width, labelSize.height + baseLine)),
            txt_bk_color, -1);

        cv::putText(image, text, cv::Point(x, y + labelSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.35 * scale, txtColor, scale);

        // Pose estimation
        if (!object.kps.empty()) {
            auto& kps = object.kps;
            for (int k = 0; k < m_numKps + 2; k++) {
                if (k < m_numKps) {
                    int   kpsX = std::round(kps[k * 3]);
                    int   kpsY = std::round(kps[k * 3 + 1]);
                    float kpsS = kps[k * 3 + 2];
                    if (kpsS > m_kpsThreshold) {
                        cv::Scalar kpsColor = cv::Scalar(KPS_COLORS[k][0], KPS_COLORS[k][1], KPS_COLORS[k][2]);
                        cv::circle(image, { kpsX, kpsY }, 5, kpsColor, -1);
                    }
                }
                auto& ske = SKELETON[k];
                int   pos1X = std::round(kps[(ske[0] - 1) * 3]);
                int   pos1Y = std::round(kps[(ske[0] - 1) * 3 + 1]);

                int pos2X = std::round(kps[(ske[1] - 1) * 3]);
                int pos2Y = std::round(kps[(ske[1] - 1) * 3 + 1]);

                float pos1S = kps[(ske[0] - 1) * 3 + 2];
                float pos2S = kps[(ske[1] - 1) * 3 + 2];

                if (pos1S > m_kpsThreshold && pos2S > m_kpsThreshold) {
                    cv::Scalar limbColor = cv::Scalar(LIMB_COLORS[k][0], LIMB_COLORS[k][1], LIMB_COLORS[k][2]);
                    cv::line(image, { pos1X, pos1Y }, { pos2X, pos2Y }, limbColor, 2);
                }
            }
        }
    }
}

void YoloV8::preprocess(const cv::cuda::GpuMat& gpuImg, std::vector<std::vector<cv::cuda::GpuMat>>& inputs)
{
    // Populate the gpuImg vectors
    const auto& inputDims = m_trtEngine->getInputDims();

    // Convert the image from BGR to RGB
    cv::cuda::GpuMat rgbMat;
    cv::cuda::cvtColor(gpuImg, rgbMat, cv::COLOR_BGR2RGB);

    auto resized = rgbMat;

    // Resize to the model expected gpuImg size while maintaining the aspect ratio with the use of padding
    if (resized.rows != inputDims[0].d[1] || resized.cols != inputDims[0].d[2]) {
        // Only resize if not already the right size to avoid unnecessary copy
        resized = Engine::resizeKeepAspectRatioPadRightBottom(rgbMat, inputDims[0].d[1], inputDims[0].d[2]);
    }

    // Convert to format expected by our inference engine
    // The reason for the strange format is because it supports models with multiple inputs as well as batching
    // In our case though, the model only has a single gpuImg and we are using a batch size of 1.
    std::vector<cv::cuda::GpuMat> input{resized};
    inputs = { std::move(input) };

    // These params will be used in the post-processing stage
    m_imgHeight = static_cast<float>(rgbMat.rows);
    m_imgWidth = static_cast<float>(rgbMat.cols);
    m_ratio = 1.f / std::min(inputDims[0].d[2] / static_cast<float>(rgbMat.cols), inputDims[0].d[1] / static_cast<float>(rgbMat.rows));
}

void YoloV8::postprocessDetection(std::vector<float>& featureVector, std::vector<InferenceObject>& inferenceObjects)
{
    const auto& outputDims = m_trtEngine->getOutputDims();
    auto numChannels = outputDims[0].d[1];
    auto numAnchors = outputDims[0].d[2];

    auto numClasses = m_classNames.size();

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
        if (score > m_probabilityThreshold) {
            float x = *bboxesPtr++;
            float y = *bboxesPtr++;
            float w = *bboxesPtr++;
            float h = *bboxesPtr;

#if HAS_CPP_17
            float x0 = std::clamp((x - 0.5f * w) * m_ratio, 0.f, m_imgWidth);
            float y0 = std::clamp((y - 0.5f * h) * m_ratio, 0.f, m_imgHeight);
            float x1 = std::clamp((x + 0.5f * w) * m_ratio, 0.f, m_imgWidth);
            float y1 = std::clamp((y + 0.5f * h) * m_ratio, 0.f, m_imgHeight);
#else
            float x0 = clamp((x - 0.5f * w) * m_ratio, 0.f, m_imgWidth);
            float y0 = clamp((y - 0.5f * h) * m_ratio, 0.f, m_imgHeight);
            float x1 = clamp((x + 0.5f * w) * m_ratio, 0.f, m_imgWidth);
            float y1 = clamp((y + 0.5f * h) * m_ratio, 0.f, m_imgHeight);
#endif

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
    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, m_probabilityThreshold, m_nmsThreshold, indices);

    // Choose the top k detections
    int cnt = 0;
    for (auto& chosenIdx : indices) {
        if (cnt >= m_topK) {
            break;
        }

        InferenceObject obj{};
        obj.probability = scores[chosenIdx];
        obj.label = labels[chosenIdx];
        obj.rect = bboxes[chosenIdx];
        inferenceObjects.push_back(obj);

        cnt += 1;
    }
}

void YoloV8::postprocessPose(std::vector<float>& featureVector, std::vector<InferenceObject>& inferenceObjects)
{
    const auto& outputDims = m_trtEngine->getOutputDims();
    auto numChannels = outputDims[0].d[1];
    auto numAnchors = outputDims[0].d[2];

    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<int> indices;
    std::vector<std::vector<float>> kpss;

    cv::Mat output = cv::Mat(numChannels, numAnchors, CV_32F, featureVector.data());
    output = output.t();

    // Get all the YOLO proposals
    for (int i = 0; i < numAnchors; i++) {
        auto rowPtr = output.row(i).ptr<float>();
        auto bboxesPtr = rowPtr;
        auto scoresPtr = rowPtr + 4;
        auto kps_ptr = rowPtr + 5;
        float score = *scoresPtr;
        if (score > m_probabilityThreshold) {
            float x = *bboxesPtr++;
            float y = *bboxesPtr++;
            float w = *bboxesPtr++;
            float h = *bboxesPtr;

#if HAS_CPP_17
            float x0 = std::clamp((x - 0.5f * w) * m_ratio, 0.f, m_imgWidth);
            float y0 = std::clamp((y - 0.5f * h) * m_ratio, 0.f, m_imgHeight);
            float x1 = std::clamp((x + 0.5f * w) * m_ratio, 0.f, m_imgWidth);
            float y1 = std::clamp((y + 0.5f * h) * m_ratio, 0.f, m_imgHeight);
#else
            float x0 = clamp((x - 0.5f * w) * m_ratio, 0.f, m_imgWidth);
            float y0 = clamp((y - 0.5f * h) * m_ratio, 0.f, m_imgHeight);
            float x1 = clamp((x + 0.5f * w) * m_ratio, 0.f, m_imgWidth);
            float y1 = clamp((y + 0.5f * h) * m_ratio, 0.f, m_imgHeight);
#endif

            cv::Rect_<float> bbox;
            bbox.x = x0;
            bbox.y = y0;
            bbox.width = x1 - x0;
            bbox.height = y1 - y0;

            std::vector<float> kps;
            for (int k = 0; k < m_numKps; k++) {
                float kpsX = *(kps_ptr + 3 * k) * m_ratio;
                float kpsY = *(kps_ptr + 3 * k + 1) * m_ratio;
                float kpsS = *(kps_ptr + 3 * k + 2);
#if HAS_CPP_17
                kpsX = std::clamp(kpsX, 0.f, m_imgWidth);
                kpsY = std::clamp(kpsY, 0.f, m_imgHeight);
#else
                kpsX = clamp(kpsX, 0.f, m_imgWidth);
                kpsY = clamp(kpsY, 0.f, m_imgHeight);
#endif
                kps.push_back(kpsX);
                kps.push_back(kpsY);
                kps.push_back(kpsS);
            }

            bboxes.push_back(bbox);
            labels.push_back(0); // All detected objects are people
            scores.push_back(score);
            kpss.push_back(kps);
        }
    }

    // Run NMS
    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, m_probabilityThreshold, m_nmsThreshold, indices);

    // Choose the top k detections
    int cnt = 0;
    for (auto& chosenIdx : indices) {
        if (cnt >= m_topK) {
            break;
        }

        InferenceObject obj{};
        obj.probability = scores[chosenIdx];
        obj.label = labels[chosenIdx];
        obj.rect = bboxes[chosenIdx];
        obj.kps = kpss[chosenIdx];
        inferenceObjects.push_back(obj);

        cnt += 1;
    }
}

void YoloV8::postprocessSegmentation(std::vector<std::vector<float>>& featureVectors,
                                     std::vector<InferenceObject>& inferenceObjects)
{
    const auto& outputDims = m_trtEngine->getOutputDims();

    int numChannels = outputDims[outputDims.size() - 1].d[1];
    int numAnchors = outputDims[outputDims.size() - 1].d[2];

    const auto numClasses = numChannels - m_segChannels - 4;

    // Ensure the output lengths are correct
    if (featureVectors[0].size() != static_cast<size_t>(m_segChannels) * m_segH * m_segW) {
        throw std::logic_error("Output at index 0 has incorrect length");
    }

    if (featureVectors[1].size() != static_cast<size_t>(numChannels) * numAnchors) {
        throw std::logic_error("Output at index 1 has incorrect length");
    }

    cv::Mat output = cv::Mat(numChannels, numAnchors, CV_32F, featureVectors[1].data());
    output = output.t();

    cv::Mat protos = cv::Mat(m_segChannels, m_segH * m_segW, CV_32F, featureVectors[0].data());

    std::vector<int> labels;
    std::vector<float> scores;
    std::vector<cv::Rect> bboxes;
    std::vector<cv::Mat> maskConfs;
    std::vector<int> indices;

    // InferenceObject the bounding boxes and class labels
    for (int i = 0; i < numAnchors; i++) {
        auto rowPtr = output.row(i).ptr<float>();
        auto bboxesPtr = rowPtr;
        auto scoresPtr = rowPtr + 4;
        auto maskConfsPtr = rowPtr + 4 + numClasses;
        auto maxSPtr = std::max_element(scoresPtr, scoresPtr + numClasses);
        float score = *maxSPtr;
        if (score > m_probabilityThreshold) {
            float x = *bboxesPtr++;
            float y = *bboxesPtr++;
            float w = *bboxesPtr++;
            float h = *bboxesPtr;

#if HAS_CPP_17
            float x0 = std::clamp((x - 0.5f * w) * m_ratio, 0.f, m_imgWidth);
            float y0 = std::clamp((y - 0.5f * h) * m_ratio, 0.f, m_imgHeight);
            float x1 = std::clamp((x + 0.5f * w) * m_ratio, 0.f, m_imgWidth);
            float y1 = std::clamp((y + 0.5f * h) * m_ratio, 0.f, m_imgHeight);
#else
            float x0 = clamp((x - 0.5f * w) * m_ratio, 0.f, m_imgWidth);
            float y0 = clamp((y - 0.5f * h) * m_ratio, 0.f, m_imgHeight);
            float x1 = clamp((x + 0.5f * w) * m_ratio, 0.f, m_imgWidth);
            float y1 = clamp((y + 0.5f * h) * m_ratio, 0.f, m_imgHeight);
#endif

            int label = maxSPtr - scoresPtr;
            cv::Rect_<float> bbox;
            bbox.x = x0;
            bbox.y = y0;
            bbox.width = x1 - x0;
            bbox.height = y1 - y0;

            cv::Mat maskConf = cv::Mat(1, m_segChannels, CV_32F, maskConfsPtr);

            bboxes.push_back(bbox);
            labels.push_back(label);
            scores.push_back(score);
            maskConfs.push_back(maskConf);
        }
    }

    // Require OpenCV 4.7 for this function
    cv::dnn::NMSBoxesBatched(
        bboxes,
        scores,
        labels,
        m_probabilityThreshold,
        m_nmsThreshold,
        indices
    );

    // Obtain the segmentation masks
    cv::Mat masks;
    int cnt = 0;
    for (auto& i : indices) {
        if (cnt >= m_topK) {
            break;
        }
        cv::Rect tmp = bboxes[i];
        InferenceObject obj;
        obj.label = labels[i];
        obj.rect = tmp;
        obj.probability = scores[i];
        masks.push_back(maskConfs[i]);
        inferenceObjects.push_back(obj);
        cnt += 1;
    }

    // Convert segmentation mask to original frame
    if (!masks.empty()) {
        cv::Mat matmulRes = (masks * protos).t();
        cv::Mat maskMat = matmulRes.reshape(indices.size(), { m_segW, m_segH });

        std::vector<cv::Mat> maskChannels;
        cv::split(maskMat, maskChannels);
        const auto inputDims = m_trtEngine->getInputDims();

        cv::Rect roi;
        if (m_imgHeight > m_imgWidth) {
            roi = cv::Rect(0, 0, m_segW * m_imgWidth / m_imgHeight, m_segH);
        }
        else {
            roi = cv::Rect(0, 0, m_segW, m_segH * m_imgHeight / m_imgWidth);
        }


        for (size_t i = 0; i < indices.size(); i++)
        {
            cv::Mat dest, mask;
            cv::exp(-maskChannels[i], dest);
            dest = 1.0 / (1.0 + dest);
            dest = dest(roi);
            cv::resize(
                dest,
                mask,
                cv::Size(static_cast<int>(m_imgWidth), static_cast<int>(m_imgHeight)),
                cv::INTER_LINEAR
            );
            inferenceObjects[i].boxMask = mask(inferenceObjects[i].rect) > m_segmentationThreshold;
        }
    }
}

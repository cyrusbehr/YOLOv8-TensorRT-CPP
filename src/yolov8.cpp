#include "yolov8.h"
#include "stopwatch.h"
#include <iostream> // std::cout in the ENABLE_BENCHMARKS timing (was transitive via the v6 engine.h)
#include <stdexcept>

namespace {
// Unwrap a v7 Result, throwing on error (this app uses exceptions). Calling .value() directly
// would assert in debug builds and be undefined behavior in -DNDEBUG release builds when the
// Result holds an error (e.g. a dynamic/oversized shape, OOM, or a non-float output dtype).
template <class T> T must(trtcpp::Result<T> r, const char *what) {
    if (!r) {
        throw std::runtime_error(std::string("Error: ") + what + ": " + r.status().message());
    }
    return std::move(r).value();
}
} // namespace

YoloV8::YoloV8(const std::string &onnxModelPath, const std::string &trtModelPath, const YoloV8Config &config)
    : PROBABILITY_THRESHOLD(config.probabilityThreshold), NMS_THRESHOLD(config.nmsThreshold), TOP_K(config.topK),
      SEG_CHANNELS(config.segChannels), SEG_H(config.segH), SEG_W(config.segW), SEGMENTATION_THRESHOLD(config.segmentationThreshold),
      CLASS_NAMES(config.classNames), NUM_KPS(config.numKPS), KPS_THRESHOLD(config.kpsThreshold) {
    // Specify build options for the v7 engine builder. (Batch knobs are now expressed as
    // optimization profiles; this detector uses the model's static 1x3xHxW input.)
    trtcpp::BuildOptions options;
    options.precision = config.precision;
    options.engineCacheDir = "."; // build-or-load caches next to the working dir; v7 detects staleness

    // v7 INT8: prefer an explicit-QDQ ONNX with Precision::kInt8Qdq (no calibration data). Legacy
    // calibrator PTQ (kInt8CalibLegacy) is only available when the library is built against
    // TensorRT < 11 and is wired via BuildOptions.calibrator (see tensorrt_cpp_api/calibrator.h).
    if (options.precision == trtcpp::Precision::kInt8CalibLegacy && config.calibrationDataDirectory.empty()) {
        throw std::runtime_error("Error: Must supply calibration data path for legacy INT8 calibration");
    }

    // Obtain a ready-to-run v7 Engine, either by building the ONNX into a TensorRT engine (caching
    // it next to the working dir, rebuilding only when stale) or by loading a prebuilt .trt/.engine
    // file directly. Preprocessing (BGR->RGB, letterbox, 1/255 scale) is fused on the GPU in
    // preprocess(), so the v6 SUB_VALS/DIV_VALS/NORMALIZE are no longer passed at build/load time.
    auto loadEngine = [&]() -> trtcpp::Result<trtcpp::Engine> {
        if (!onnxModelPath.empty()) {
            // Build the ONNX model into a TensorRT engine (or load a fresh cached one) and deserialize it.
            return trtcpp::EngineBuilder{}.buildAndLoad(onnxModelPath, options);
        }
        if (!trtModelPath.empty()) {
            // No ONNX model: deserialize a prebuilt TensorRT engine file directly.
            return trtcpp::Engine::loadFromFile(trtModelPath);
        }
        return trtcpp::Status{trtcpp::StatusCode::kInvalidArgument, "Neither ONNX model nor TensorRT engine path provided."};
    };
    auto engine = loadEngine();
    if (!engine) {
        throw std::runtime_error("Error: Unable to build or load the TensorRT engine: " + engine.status().message());
    }
    m_engine = std::make_unique<trtcpp::Engine>(std::move(engine).value());

    // Cache IO metadata once (v7 is name-keyed and non-templated).
    m_inputName = m_engine->inputNames().front();
    m_outputNames = m_engine->outputNames();
    m_inputShape = must(m_engine->tensorShape(m_inputName), "query input shape"); // [1,3,H,W]
    for (const auto &name : m_outputNames) {
        m_outputShapes.push_back(must(m_engine->tensorShape(name), "query output shape"));
    }

    // Pre-allocate the NCHW float input tensor. allocate() errors (and we throw) on a dynamic
    // input shape or a CUDA OOM rather than crashing on an unchecked .value().
    m_input = must(trtcpp::Tensor::allocate(trtcpp::DType::kFloat32, m_inputShape, trtcpp::Device::kCuda), "allocate input tensor");
}

void YoloV8::preprocess(const cv::cuda::GpuMat &gpuImg) {
    // Record original dims + the letterbox ratio used by post-processing to map boxes back to the
    // source image. inputShape is [1, 3, H, W].
    m_imgHeight = static_cast<float>(gpuImg.rows);
    m_imgWidth = static_cast<float>(gpuImg.cols);
    const int inH = static_cast<int>(m_inputShape[2]);
    const int inW = static_cast<int>(m_inputShape[3]);
    m_ratio = 1.f / std::min(inW / m_imgWidth, inH / m_imgHeight);

    // One fused GPU kernel replaces the v6 cvtColor + resizeKeepAspectRatioPadRightBottom and the
    // in-engine HWC->NCHW + normalize: BGR->RGB, letterbox-resize (pad right/bottom), scale by
    // 1/255 (SUB_VALS=0, DIV_VALS=1, NORMALIZE), and write the NCHW float input tensor in place.
    trtcpp::preproc::PreprocSpec spec;
    spec.swapRB = true;             // OpenCV GpuMat is BGR; the model expects RGB
    spec.keepAspectRatioPad = true; // letterbox, pad right/bottom (matches v6)
    spec.scale = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f, 1.f};

    // cv::cuda::GpuMat rows are typically pitched (padded for alignment) and a TensorView is
    // contiguous, so make a continuous copy when the upload isn't already continuous.
    cv::cuda::GpuMat continuous = gpuImg;
    if (!gpuImg.isContinuous()) {
        cv::cuda::createContinuous(gpuImg.rows, gpuImg.cols, gpuImg.type(), continuous);
        gpuImg.copyTo(continuous);
    }
    auto src = trtcpp::opencv::viewOf(continuous); // zero-copy HWC-uint8 device view
    if (!src) {
        throw std::runtime_error("Error: could not view the input GpuMat: " + src.status().message());
    }
    if (auto s = trtcpp::preproc::letterboxToTensor(src.value(), m_input.view(), spec, m_stream); !s) {
        throw std::runtime_error("Error: preprocessing failed: " + s.message());
    }
}

std::vector<Object> YoloV8::detectObjects(const cv::cuda::GpuMat &inputImageBGR) {
    // Preprocess the input image
#ifdef ENABLE_BENCHMARKS
    static int numIts = 1;
    preciseStopwatch s1;
#endif
    preprocess(inputImageBGR); // fills m_input
#ifdef ENABLE_BENCHMARKS
    static long long t1 = 0;
    t1 += s1.elapsedTime<long long, std::chrono::microseconds>();
    std::cout << "Avg Preprocess time: " << (t1 / numIts) / 1000.f << " ms" << std::endl;
#endif
    // Run inference using the TensorRT engine
#ifdef ENABLE_BENCHMARKS
    preciseStopwatch s2;
#endif
    auto outputs = m_engine->infer({{m_inputName, m_input.view()}}, m_stream);
    if (!outputs) {
        throw std::runtime_error("Error: Unable to run inference: " + outputs.status().message());
    }
    // Read each output back to a flat host float vector, in output-binding order. (v7 returns
    // name-keyed owning Tensors; toHost performs the D2H copy AND synchronizes the stream.)
    std::vector<std::vector<float>> featureVectors;
    featureVectors.reserve(m_outputNames.size());
    for (const auto &name : m_outputNames) {
        auto host = outputs->at(name).toHost(m_stream);
        if (!host) {
            throw std::runtime_error("Error: output readback failed: " + host.status().message());
        }
        const auto span = must(host->as<float>(), "output tensor is not float32 (rebuild the engine with a float output)");
        featureVectors.emplace_back(span.begin(), span.end());
    }
#ifdef ENABLE_BENCHMARKS
    static long long t2 = 0;
    t2 += s2.elapsedTime<long long, std::chrono::microseconds>();
    std::cout << "Avg Inference time: " << (t2 / numIts) / 1000.f << " ms" << std::endl;
    preciseStopwatch s3;
#endif
    // Check if our model does only object detection or also supports segmentation
    // v7 already gives one flat host vector per output (batch size 1), so the v6 transformOutput
    // 3D->1D/2D flattening is no longer needed.
    std::vector<Object> ret;
    if (m_outputShapes.size() == 1) {
        // Object detection or pose estimation. Output shape is [1, C, anchors]; the channel count C
        // distinguishes the two: pose adds NUM_KPS*3 keypoint values on top of (4 box + classes),
        // while plain detection is just (4 box + classes). No magic number; works with Ultralytics
        // pretrained models.
        const size_t numChannels = static_cast<size_t>(m_outputShapes[0][1]);
        if (numChannels == 4 + CLASS_NAMES.size() + NUM_KPS * 3) {
            // Pose estimation
            ret = postprocessPose(featureVectors[0]);
        } else if (numChannels == 4 + CLASS_NAMES.size()) {
            // Object detection
            ret = postprocessDetect(featureVectors[0]);
        }
        else {
            throw std::runtime_error("Error: Unable to identify whether the model is for Pose estimation or Object detection.");
        }
    } else {
        // Instance segmentation (detections + mask prototypes).
        ret = postProcessSegmentation(featureVectors);
    }
#ifdef ENABLE_BENCHMARKS
    static long long t3 = 0;
    t3 += s3.elapsedTime<long long, std::chrono::microseconds>();
    std::cout << "Avg Postprocess time: " << (t3 / numIts++) / 1000.f << " ms\n" << std::endl;
#endif
    return ret;
}

std::vector<Object> YoloV8::detectObjects(const cv::Mat &inputImageBGR) {
    // Upload the image to GPU memory
    cv::cuda::GpuMat gpuImg;
    gpuImg.upload(inputImageBGR);

    // Call detectObjects with the GPU image
    return detectObjects(gpuImg);
}

std::vector<Object> YoloV8::postProcessSegmentation(std::vector<std::vector<float>> &featureVectors) {
    int numChannels = static_cast<int>(m_outputShapes[0][1]);
    int numAnchors = static_cast<int>(m_outputShapes[0][2]);

    const auto numClasses = numChannels - SEG_CHANNELS - 4;

    // Ensure the output lengths are correct
    if (featureVectors[0].size() != static_cast<size_t>(numChannels) * numAnchors) {
        throw std::logic_error("Output at index 0 has incorrect length");
    }

    if (featureVectors[1].size() != static_cast<size_t>(SEG_CHANNELS) * SEG_H * SEG_W) {
        throw std::logic_error("Output at index 1 has incorrect length");
    }

    cv::Mat output = cv::Mat(numChannels, numAnchors, CV_32F, featureVectors[0].data());
    output = output.t();

    cv::Mat protos = cv::Mat(SEG_CHANNELS, SEG_H * SEG_W, CV_32F, featureVectors[1].data());

    std::vector<int> labels;
    std::vector<float> scores;
    std::vector<cv::Rect> bboxes;
    std::vector<cv::Mat> maskConfs;
    std::vector<int> indices;

    // Object the bounding boxes and class labels
    for (int i = 0; i < numAnchors; i++) {
        auto rowPtr = output.row(i).ptr<float>();
        auto bboxesPtr = rowPtr;
        auto scoresPtr = rowPtr + 4;
        auto maskConfsPtr = rowPtr + 4 + numClasses;
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

            cv::Mat maskConf = cv::Mat(1, SEG_CHANNELS, CV_32F, maskConfsPtr);

            bboxes.push_back(bbox);
            labels.push_back(label);
            scores.push_back(score);
            maskConfs.push_back(maskConf);
        }
    }

    // Require OpenCV 4.7 for this function
    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, PROBABILITY_THRESHOLD, NMS_THRESHOLD, indices);

    // Obtain the segmentation masks
    cv::Mat masks;
    std::vector<Object> objs;
    int cnt = 0;
    for (auto &i : indices) {
        if (cnt >= TOP_K) {
            break;
        }
        cv::Rect tmp = bboxes[i];
        Object obj;
        obj.label = labels[i];
        obj.rect = tmp;
        obj.probability = scores[i];
        masks.push_back(maskConfs[i]);
        objs.push_back(obj);
        cnt += 1;
    }

    // Convert segmentation mask to original frame
    if (!masks.empty()) {
        cv::Mat matmulRes = (masks * protos).t();
        cv::Mat maskMat = matmulRes.reshape(indices.size(), {SEG_W, SEG_H});

        std::vector<cv::Mat> maskChannels;
        cv::split(maskMat, maskChannels);

        cv::Rect roi;
        if (m_imgHeight > m_imgWidth) {
            roi = cv::Rect(0, 0, SEG_W * m_imgWidth / m_imgHeight, SEG_H);
        } else {
            roi = cv::Rect(0, 0, SEG_W, SEG_H * m_imgHeight / m_imgWidth);
        }

        for (size_t i = 0; i < indices.size(); i++) {
            cv::Mat dest, mask;
            cv::exp(-maskChannels[i], dest);
            dest = 1.0 / (1.0 + dest);
            dest = dest(roi);
            cv::resize(dest, mask, cv::Size(static_cast<int>(m_imgWidth), static_cast<int>(m_imgHeight)), cv::INTER_LINEAR);
            objs[i].boxMask = mask(objs[i].rect) > SEGMENTATION_THRESHOLD;
        }
    }

    return objs;
}

std::vector<Object> YoloV8::postprocessPose(std::vector<float> &featureVector) {
    const auto numChannels = static_cast<int>(m_outputShapes[0][1]);
    const auto numAnchors = static_cast<int>(m_outputShapes[0][2]);

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
        if (score > PROBABILITY_THRESHOLD) {
            float x = *bboxesPtr++;
            float y = *bboxesPtr++;
            float w = *bboxesPtr++;
            float h = *bboxesPtr;

            float x0 = std::clamp((x - 0.5f * w) * m_ratio, 0.f, m_imgWidth);
            float y0 = std::clamp((y - 0.5f * h) * m_ratio, 0.f, m_imgHeight);
            float x1 = std::clamp((x + 0.5f * w) * m_ratio, 0.f, m_imgWidth);
            float y1 = std::clamp((y + 0.5f * h) * m_ratio, 0.f, m_imgHeight);

            cv::Rect_<float> bbox;
            bbox.x = x0;
            bbox.y = y0;
            bbox.width = x1 - x0;
            bbox.height = y1 - y0;

            std::vector<float> kps;
            for (int k = 0; k < NUM_KPS; k++) {
                float kpsX = *(kps_ptr + 3 * k) * m_ratio;
                float kpsY = *(kps_ptr + 3 * k + 1) * m_ratio;
                float kpsS = *(kps_ptr + 3 * k + 2);
                kpsX = std::clamp(kpsX, 0.f, m_imgWidth);
                kpsY = std::clamp(kpsY, 0.f, m_imgHeight);
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
    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, PROBABILITY_THRESHOLD, NMS_THRESHOLD, indices);

    std::vector<Object> objects;

    // Choose the top k detections
    int cnt = 0;
    for (auto &chosenIdx : indices) {
        if (cnt >= TOP_K) {
            break;
        }

        Object obj{};
        obj.probability = scores[chosenIdx];
        obj.label = labels[chosenIdx];
        obj.rect = bboxes[chosenIdx];
        obj.kps = kpss[chosenIdx];
        objects.push_back(obj);

        cnt += 1;
    }

    return objects;
}

std::vector<Object> YoloV8::postprocessDetect(std::vector<float> &featureVector) {
    const auto numChannels = static_cast<int>(m_outputShapes[0][1]);
    const auto numAnchors = static_cast<int>(m_outputShapes[0][2]);

    auto numClasses = CLASS_NAMES.size();

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
    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, PROBABILITY_THRESHOLD, NMS_THRESHOLD, indices);

    std::vector<Object> objects;

    // Choose the top k detections
    int cnt = 0;
    for (auto &chosenIdx : indices) {
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

void YoloV8::drawObjectLabels(cv::Mat &image, const std::vector<Object> &objects, unsigned int scale) {
    // If segmentation information is present, start with that
    if (!objects.empty() && !objects[0].boxMask.empty()) {
        cv::Mat mask = image.clone();
        for (const auto &object : objects) {
            // Choose the color
            int colorIndex = object.label % COLOR_LIST.size(); // We have only defined 80 unique colors
            cv::Scalar color = cv::Scalar(COLOR_LIST[colorIndex][0], COLOR_LIST[colorIndex][1], COLOR_LIST[colorIndex][2]);

            // Add the mask for said object
            mask(object.rect).setTo(color * 255, object.boxMask);
        }
        // Add all the masks to our image
        cv::addWeighted(image, 0.5, mask, 0.8, 1, image);
    }

    // Bounding boxes and annotations
    for (auto &object : objects) {
        // Choose the color
        int colorIndex = object.label % COLOR_LIST.size(); // We have only defined 80 unique colors
        cv::Scalar color = cv::Scalar(COLOR_LIST[colorIndex][0], COLOR_LIST[colorIndex][1], COLOR_LIST[colorIndex][2]);
        float meanColor = cv::mean(color)[0];
        cv::Scalar txtColor;
        if (meanColor > 0.5) {
            txtColor = cv::Scalar(0, 0, 0);
        } else {
            txtColor = cv::Scalar(255, 255, 255);
        }

        const auto &rect = object.rect;

        // Draw rectangles and text
        char text[256];
        sprintf(text, "%s %.1f%%", CLASS_NAMES[object.label].c_str(), object.probability * 100);

        int baseLine = 0;
        cv::Size labelSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.35 * scale, scale, &baseLine);

        cv::Scalar txt_bk_color = color * 0.7 * 255;

        int x = object.rect.x;
        int y = object.rect.y + 1;

        cv::rectangle(image, rect, color * 255, scale + 1);

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(labelSize.width, labelSize.height + baseLine)), txt_bk_color, -1);

        cv::putText(image, text, cv::Point(x, y + labelSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.35 * scale, txtColor, scale);

        // Pose estimation
        if (!object.kps.empty()) {
            auto &kps = object.kps;
            for (int k = 0; k < NUM_KPS + 2; k++) {
                if (k < NUM_KPS) {
                    int kpsX = std::round(kps[k * 3]);
                    int kpsY = std::round(kps[k * 3 + 1]);
                    float kpsS = kps[k * 3 + 2];
                    if (kpsS > KPS_THRESHOLD) {
                        cv::Scalar kpsColor = cv::Scalar(KPS_COLORS[k][0], KPS_COLORS[k][1], KPS_COLORS[k][2]);
                        cv::circle(image, {kpsX, kpsY}, 5, kpsColor, -1);
                    }
                }
                auto &ske = SKELETON[k];
                int pos1X = std::round(kps[(ske[0] - 1) * 3]);
                int pos1Y = std::round(kps[(ske[0] - 1) * 3 + 1]);

                int pos2X = std::round(kps[(ske[1] - 1) * 3]);
                int pos2Y = std::round(kps[(ske[1] - 1) * 3 + 1]);

                float pos1S = kps[(ske[0] - 1) * 3 + 2];
                float pos2S = kps[(ske[1] - 1) * 3 + 2];

                if (pos1S > KPS_THRESHOLD && pos2S > KPS_THRESHOLD) {
                    cv::Scalar limbColor = cv::Scalar(LIMB_COLORS[k][0], LIMB_COLORS[k][1], LIMB_COLORS[k][2]);
                    cv::line(image, {pos1X, pos1Y}, {pos2X, pos2Y}, limbColor, 2);
                }
            }
        }
    }
}

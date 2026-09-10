#include "s3m/detection/onnx_detector.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>

#include "s3m/detection/letterbox.hpp"
#include "s3m/detection/nms.hpp"

namespace s3m {

// NOTE: model_path is passed as `const char*` (ORTCHAR_T on Linux). A Windows
// build would need the wide-char overload; the project builds Linux-only.
struct OnnxDetector::Impl {
    Impl() : env(ORT_LOGGING_LEVEL_WARNING, "s3m") {}

    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::AllocatorWithDefaultOptions allocator;
    std::unique_ptr<Ort::Session> session;

    std::string input_name;
    std::vector<std::string> output_names;
};

namespace {

/// Letterboxed BGR image -> CHW, RGB, [0,1] float blob.
std::vector<float> toBlob(const cv::Mat& letterboxed_bgr, int size) {
    cv::Mat rgb;
    cv::cvtColor(letterboxed_bgr, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    const std::size_t plane = static_cast<std::size_t>(size) * static_cast<std::size_t>(size);
    std::vector<float> blob(3 * plane);
    std::vector<cv::Mat> channels{
        cv::Mat(size, size, CV_32F, blob.data() + 0 * plane),
        cv::Mat(size, size, CV_32F, blob.data() + 1 * plane),
        cv::Mat(size, size, CV_32F, blob.data() + 2 * plane),
    };
    cv::split(rgb, channels);
    return blob;
}

}  // namespace

OnnxDetector::OnnxDetector(Options options) : options_(std::move(options)) {
    if (options_.model_path.empty()) {
        throw std::invalid_argument("OnnxDetector: model_path is empty");
    }
    if (options_.input_size <= 0) {
        throw std::invalid_argument("OnnxDetector: input_size must be positive");
    }

    impl_ = std::make_unique<Impl>();
    if (options_.num_threads > 0) {
        impl_->session_options.SetIntraOpNumThreads(options_.num_threads);
    }
    impl_->session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    impl_->session = std::make_unique<Ort::Session>(impl_->env, options_.model_path.c_str(),
                                                    impl_->session_options);

    if (impl_->session->GetInputCount() != 1) {
        throw std::runtime_error("OnnxDetector: model must have exactly one input");
    }
    impl_->input_name = impl_->session->GetInputNameAllocated(0, impl_->allocator).get();

    const std::size_t outputs = impl_->session->GetOutputCount();
    for (std::size_t i = 0; i < outputs; ++i) {
        impl_->output_names.emplace_back(
            impl_->session->GetOutputNameAllocated(i, impl_->allocator).get());
    }
    if (impl_->output_names.empty()) {
        throw std::runtime_error("OnnxDetector: model has no outputs");
    }
}

OnnxDetector::~OnnxDetector() = default;
OnnxDetector::OnnxDetector(OnnxDetector&&) noexcept = default;
OnnxDetector& OnnxDetector::operator=(OnnxDetector&&) noexcept = default;

std::vector<Detection2D> OnnxDetector::detect(const cv::Mat& image_bgr) {
    CV_Assert(!image_bgr.empty());

    cv::Mat bgr = image_bgr;
    if (image_bgr.channels() == 1) cv::cvtColor(image_bgr, bgr, cv::COLOR_GRAY2BGR);

    const int size = options_.input_size;
    cv::Mat lb_img;
    const Letterbox lb = makeLetterbox(bgr, lb_img, size);
    std::vector<float> blob = toBlob(lb_img, size);

    const std::array<std::int64_t, 4> in_shape{1, 3, size, size};
    const Ort::MemoryInfo mem =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input = Ort::Value::CreateTensor<float>(mem, blob.data(), blob.size(),
                                                       in_shape.data(), in_shape.size());

    const char* input_names[] = {impl_->input_name.c_str()};
    std::vector<const char*> output_names;
    output_names.reserve(impl_->output_names.size());
    for (const std::string& n : impl_->output_names) output_names.push_back(n.c_str());

    const std::vector<Ort::Value> outputs =
        impl_->session->Run(Ort::RunOptions{nullptr}, input_names, &input, 1, output_names.data(),
                            output_names.size());

    const float* data = outputs[0].GetTensorData<float>();
    const std::vector<std::int64_t> shape =
        outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 3 || shape[0] != 1) {
        throw std::runtime_error("OnnxDetector: expected a rank-3 [1, *, *] output");
    }

    // YOLOv8: [1, 4+nc, anchors] (attrs < anchors). YOLOv5: [1, anchors, 5+nc].
    const bool v8 = shape[1] < shape[2];
    const std::int64_t num_anchors = v8 ? shape[2] : shape[1];
    const std::int64_t attrs = v8 ? shape[1] : shape[2];
    const std::int64_t class_off = v8 ? 4 : 5;
    const std::int64_t num_classes = attrs - class_off;
    if (num_classes <= 0) {
        throw std::runtime_error("OnnxDetector: output has no class scores");
    }

    const auto at = [&](std::int64_t anchor, std::int64_t attr) -> float {
        return v8 ? data[attr * num_anchors + anchor] : data[anchor * attrs + attr];
    };

    std::vector<cv::Rect2f> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    for (std::int64_t a = 0; a < num_anchors; ++a) {
        const float obj = v8 ? 1.0f : at(a, 4);
        if (obj <= 0.0f) continue;

        int best_class = 0;
        float best_score = 0.0f;
        for (std::int64_t c = 0; c < num_classes; ++c) {
            const float s = at(a, class_off + c) * obj;
            if (s > best_score) {
                best_score = s;
                best_class = static_cast<int>(c);
            }
        }
        if (best_score < options_.score_threshold) continue;
        if (!options_.keep_classes.empty() &&
            std::find(options_.keep_classes.begin(), options_.keep_classes.end(), best_class) ==
                options_.keep_classes.end()) {
            continue;
        }

        const float cx = at(a, 0);
        const float cy = at(a, 1);
        const float w = at(a, 2);
        const float h = at(a, 3);
        boxes.emplace_back(cx - 0.5f * w, cy - 0.5f * h, w, h);
        scores.push_back(best_score);
        class_ids.push_back(best_class);
    }

    const std::vector<int> keep = nmsClassAware(boxes, scores, class_ids, options_.nms_iou,
                                                options_.score_threshold, options_.max_detections);

    const float w_max = static_cast<float>(bgr.cols);
    const float h_max = static_cast<float>(bgr.rows);
    std::vector<Detection2D> result;
    result.reserve(keep.size());
    for (const int idx : keep) {
        const std::size_t i = static_cast<std::size_t>(idx);
        const cv::Rect2f o = lb.toOriginal(boxes[i]);
        const float x0 = std::clamp(o.x, 0.0f, w_max);
        const float y0 = std::clamp(o.y, 0.0f, h_max);
        const float x1 = std::clamp(o.x + o.width, 0.0f, w_max);
        const float y1 = std::clamp(o.y + o.height, 0.0f, h_max);

        Detection2D det;
        det.box = cv::Rect2f(x0, y0, std::max(0.0f, x1 - x0), std::max(0.0f, y1 - y0));
        det.score = scores[i];
        det.class_id = class_ids[i];
        result.push_back(det);
    }
    return result;
}

}  // namespace s3m

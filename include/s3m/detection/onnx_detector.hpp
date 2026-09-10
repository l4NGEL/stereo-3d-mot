#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "s3m/detection/detector.hpp"

namespace s3m {

/// Object detector backed by ONNX Runtime (CPU execution provider).
///
/// Consumes a single-input YOLO-style ONNX model and auto-detects the output
/// layout:
///   * YOLOv8   `[1, 4 + num_classes, num_anchors]`  (no objectness)
///   * YOLOv5   `[1, num_anchors, 5 + num_classes]`  (objectness * class score)
///
/// Only built when the project is configured with `-DS3M_WITH_ONNX=ON` and ONNX
/// Runtime is found (`S3M_WITH_ONNX` is then defined). The header itself is
/// always includable — the ONNX Runtime API is hidden behind a PIMPL.
class OnnxDetector : public Detector {
 public:
    struct Options {
        std::string model_path;
        int input_size = 640;             ///< square network input
        float score_threshold = 0.25f;
        float nms_iou = 0.45f;
        int num_threads = 0;              ///< 0 -> ONNX Runtime default
        std::vector<int> keep_classes;   ///< empty -> keep every class
        int max_detections = 300;
    };

    explicit OnnxDetector(Options options);
    ~OnnxDetector() override;

    OnnxDetector(OnnxDetector&&) noexcept;
    OnnxDetector& operator=(OnnxDetector&&) noexcept;
    OnnxDetector(const OnnxDetector&) = delete;
    OnnxDetector& operator=(const OnnxDetector&) = delete;

    std::vector<Detection2D> detect(const cv::Mat& image_bgr) override;
    std::string name() const override { return "onnx"; }

    const Options& options() const { return options_; }

 private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Options options_;
};

}  // namespace s3m

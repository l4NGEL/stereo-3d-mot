#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>

#include "s3m/core/types.hpp"

namespace s3m {

/// Interface for a 2D object detector operating on the (left) image.
class Detector {
 public:
    virtual ~Detector() = default;
    virtual std::vector<Detection2D> detect(const cv::Mat& image_bgr) = 0;
    virtual std::string name() const = 0;
};

/// Always returns nothing -- pipeline placeholder / depth-only runs.
class NullDetector : public Detector {
 public:
    std::vector<Detection2D> detect(const cv::Mat& /*image_bgr*/) override { return {}; }
    std::string name() const override { return "null"; }
};

/// OpenCV HOG + linear-SVM pedestrian detector (people, class_id = 0).
/// No external model file; a stand-in until the ONNX detector lands in Phase 2.
class HogPeopleDetector : public Detector {
 public:
    struct Options {
        double hit_threshold = 0.0;
        double scale = 1.05;
        cv::Size win_stride = cv::Size(8, 8);
        cv::Size padding = cv::Size(8, 8);
        bool use_meanshift_grouping = false;
    };

    HogPeopleDetector() : HogPeopleDetector(Options{}) {}
    explicit HogPeopleDetector(Options options);

    std::vector<Detection2D> detect(const cv::Mat& image_bgr) override;
    std::string name() const override { return "hog_people"; }

 private:
    Options options_;
    cv::HOGDescriptor hog_;
};

}  // namespace s3m

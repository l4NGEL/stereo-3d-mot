#include "s3m/detection/detector.hpp"

#include <utility>

#include <opencv2/imgproc.hpp>

namespace s3m {

HogPeopleDetector::HogPeopleDetector(Options options) : options_(std::move(options)) {
    hog_.winSize = cv::Size(64, 128);
    hog_.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
}

std::vector<Detection2D> HogPeopleDetector::detect(const cv::Mat& image_bgr) {
    cv::Mat gray;
    if (image_bgr.channels() == 3) {
        cv::cvtColor(image_bgr, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image_bgr;
    }

    std::vector<cv::Rect> boxes;
    std::vector<double> weights;
    hog_.detectMultiScale(gray, boxes, weights, options_.hit_threshold, options_.win_stride,
                          options_.padding, options_.scale, 2.0, options_.use_meanshift_grouping);

    std::vector<Detection2D> detections;
    detections.reserve(boxes.size());
    for (std::size_t i = 0; i < boxes.size(); ++i) {
        Detection2D det;
        det.box = cv::Rect2f(static_cast<float>(boxes[i].x), static_cast<float>(boxes[i].y),
                             static_cast<float>(boxes[i].width),
                             static_cast<float>(boxes[i].height));
        det.score = static_cast<float>(i < weights.size() ? weights[i] : 0.0);
        det.class_id = 0;
        detections.push_back(det);
    }
    return detections;
}

}  // namespace s3m

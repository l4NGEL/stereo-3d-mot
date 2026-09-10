#include "s3m/detection/letterbox.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace s3m {

cv::Rect2f Letterbox::toOriginal(const cv::Rect2f& box) const {
    const float inv = (scale > 0.0f) ? 1.0f / scale : 1.0f;
    return {(box.x - static_cast<float>(pad_x)) * inv, (box.y - static_cast<float>(pad_y)) * inv,
            box.width * inv, box.height * inv};
}

cv::Point2f Letterbox::toOriginal(const cv::Point2f& p) const {
    const float inv = (scale > 0.0f) ? 1.0f / scale : 1.0f;
    return {(p.x - static_cast<float>(pad_x)) * inv, (p.y - static_cast<float>(pad_y)) * inv};
}

Letterbox makeLetterbox(const cv::Mat& src, cv::Mat& dst, int target,
                        const cv::Scalar& pad_value) {
    CV_Assert(!src.empty() && target > 0);

    Letterbox lb;
    lb.target = cv::Size(target, target);
    lb.scale = std::min(static_cast<float>(target) / static_cast<float>(src.cols),
                        static_cast<float>(target) / static_cast<float>(src.rows));
    lb.scaled = cv::Size(
        std::max(1, static_cast<int>(std::lround(static_cast<float>(src.cols) * lb.scale))),
        std::max(1, static_cast<int>(std::lround(static_cast<float>(src.rows) * lb.scale))));
    lb.pad_x = (target - lb.scaled.width) / 2;
    lb.pad_y = (target - lb.scaled.height) / 2;

    cv::Mat resized;
    cv::resize(src, resized, lb.scaled, 0.0, 0.0, cv::INTER_LINEAR);

    dst.create(target, target, src.type());
    dst.setTo(pad_value);
    resized.copyTo(dst(cv::Rect(lb.pad_x, lb.pad_y, lb.scaled.width, lb.scaled.height)));
    return lb;
}

}  // namespace s3m

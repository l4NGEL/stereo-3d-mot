#pragma once

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include "s3m/core/config.hpp"

namespace s3m {

/// Unifies OpenCV's StereoBM and StereoSGBM behind one call and returns a
/// floating-point disparity map in pixels (the internal fixed-point /16 is
/// already undone). Unmatched pixels are set to kInvalidDisparity.
class StereoMatcher {
 public:
    static constexpr float kInvalidDisparity = -1.0f;

    explicit StereoMatcher(const StereoMatcherParams& params = {});

    /// Rebuild the underlying matcher (call after changing params).
    void configure(const StereoMatcherParams& params);
    const StereoMatcherParams& params() const { return params_; }

    /// `left`, `right`: rectified 8-bit images (colour is converted to gray).
    /// Returns CV_32F, input size, disparity in pixels; kInvalidDisparity where
    /// no match was found.
    cv::Mat computeDisparity(const cv::Mat& left, const cv::Mat& right);

    /// Boolean mask (CV_8U, 255 = valid) for a disparity map from this class.
    static cv::Mat validMask(const cv::Mat& disparity);

 private:
    StereoMatcherParams params_;
    cv::Ptr<cv::StereoMatcher> matcher_;
};

}  // namespace s3m

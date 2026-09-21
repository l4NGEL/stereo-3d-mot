#pragma once

#include <iosfwd>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace s3m {

/// Error metrics for an estimated disparity/depth map against ground truth,
/// evaluated only over pixels valid in both maps.
///
/// Units follow the inputs: pixels for disparity maps, metres for depth maps.
struct DepthMetrics {
    double rmse = 0.0;
    double mae = 0.0;
    double abs_rel = 0.0;  ///< mean(|est - gt| / gt)
    double sq_rel = 0.0;   ///< mean((est - gt)^2 / gt)

    std::vector<double> bad_thresholds;  ///< error thresholds that were requested
    std::vector<double> bad_fraction;    ///< fraction exceeding each threshold, [0, 1]

    double delta1 = 0.0;  ///< fraction with max(est/gt, gt/est) < 1.25
    double delta2 = 0.0;  ///< ... < 1.25^2
    double delta3 = 0.0;  ///< ... < 1.25^3

    double density = 0.0;  ///< evaluated_pixels / gt_valid_pixels
    int evaluated_pixels = 0;
    int gt_valid_pixels = 0;

    std::string toString() const;
};

std::ostream& operator<<(std::ostream& os, const DepthMetrics& m);

/// Compare `estimate` against `ground_truth` (same size, single channel, CV_32F
/// or convertible). A pixel contributes only when both maps are finite and
/// inside [valid_min, valid_max].
DepthMetrics evaluate(const cv::Mat& estimate, const cv::Mat& ground_truth, double valid_min,
                      double valid_max,
                      const std::vector<double>& bad_thresholds = {1.0, 2.0, 4.0});

}  // namespace s3m

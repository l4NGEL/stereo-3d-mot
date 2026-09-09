#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "s3m/camera/stereo_rig.hpp"
#include "s3m/core/types.hpp"

namespace s3m {

/// Disparity map (pixels, CV_32F; invalid = StereoMatcher::kInvalidDisparity or
/// any value with d + doffs <= 0) -> point-cloud image (CV_32FC3) in the
/// left-camera frame. Invalid pixels become (0, 0, 0).
/// If `valid_mask` is non-null it is filled with a CV_8U mask (255 = valid).
cv::Mat reproject(const cv::Mat& disparity, const StereoRig& rig, cv::Mat* valid_mask = nullptr);

/// As above but with an explicit reprojection matrix Q.
cv::Mat reprojectWithQ(const cv::Mat& disparity, const cv::Matx44d& Q,
                       cv::Mat* valid_mask = nullptr);

/// Per-pixel depth Z [m] (CV_32F) from a disparity map. Invalid pixels -> 0.
cv::Mat disparityToDepthMap(const cv::Mat& disparity, const StereoRig& rig);

/// Robust depth [m] inside an image ROI: the median of valid depths over the
/// ROI after shrinking it toward its centre by `shrink` (0..0.5). Returns a
/// negative value when fewer than `min_valid_fraction` of the ROI pixels are
/// valid.
float robustDepthInRoi(const cv::Mat& depth_map, const cv::Rect& roi, float shrink = 0.15f,
                       float min_valid_fraction = 0.1f);

/// Promote 2D detections to 3D using a depth map and the rig geometry. Boxes
/// with insufficient valid depth are returned with `valid == false`.
std::vector<Detection3D> promoteTo3D(const std::vector<Detection2D>& detections,
                                     const cv::Mat& depth_map, const StereoRig& rig);

}  // namespace s3m

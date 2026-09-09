#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "s3m/core/types.hpp"

namespace s3m {

/// Colourise a disparity map (pixels); invalid pixels -> black.
cv::Mat colorizeDisparity(const cv::Mat& disparity, float max_disparity,
                          int colormap = cv::COLORMAP_TURBO);

/// Colourise a depth map [m] clamped to [min_m, max_m]; invalid -> black.
cv::Mat colorizeDepth(const cv::Mat& depth, float min_m, float max_m,
                      int colormap = cv::COLORMAP_TURBO);

/// Tile images into a grid; tiles are resized to the first image's size.
cv::Mat tile(const std::vector<cv::Mat>& images, int cols, int pad = 4,
             const cv::Scalar& pad_color = cv::Scalar::all(32));

/// Draw labelled 2D detections on a copy of `image_bgr`.
cv::Mat drawDetections(const cv::Mat& image_bgr, const std::vector<Detection2D>& detections);

/// Draw 3D detections (box + "id: Z.zz m") on a copy of `image_bgr`.
cv::Mat drawDetections3D(const cv::Mat& image_bgr, const std::vector<Detection3D>& detections);

}  // namespace s3m

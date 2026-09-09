#pragma once

#include <string>

#include <opencv2/core.hpp>

namespace s3m {

/// Write a coloured point cloud as an ASCII PLY file.
///   points : CV_32FC3, left-camera frame (metres).
///   colors : CV_8UC3 BGR, same size as `points`, or empty for uncoloured output.
///   mask   : CV_8U, 255 = keep the pixel; empty = keep every finite, Z > 0 point.
///   stride : keep every `stride`-th pixel in x and y (>= 1) to subsample.
/// Throws std::runtime_error if the file cannot be opened.
void writePointCloudPly(const std::string& path, const cv::Mat& points,
                        const cv::Mat& colors = cv::Mat(), const cv::Mat& mask = cv::Mat(),
                        int stride = 1);

}  // namespace s3m

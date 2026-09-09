#pragma once

#include <string>

#include <opencv2/core.hpp>

namespace s3m {

/// Read a PFM (Portable Float Map) file.
///   "Pf" -> CV_32FC1, "PF" -> CV_32FC3 (BGR order).
/// Handles both endiannesses and the format's bottom-to-top row order.
/// Middlebury stores "inf" for unknown disparities; those stay as +inf.
/// Throws std::runtime_error on malformed input.
cv::Mat readPfm(const std::string& path);

/// Write a CV_32FC1 matrix as a little-endian single-channel PFM.
void writePfm(const std::string& path, const cv::Mat& image);

}  // namespace s3m

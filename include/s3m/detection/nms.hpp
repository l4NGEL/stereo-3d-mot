#pragma once

#include <vector>

#include <opencv2/core.hpp>

namespace s3m {

/// Intersection-over-union of two axis-aligned boxes. 0 when they do not overlap
/// or either has non-positive area.
float iou(const cv::Rect2f& a, const cv::Rect2f& b);

/// Greedy non-maximum suppression.
///
/// Candidates with score < `score_threshold` are dropped, the rest are ranked by
/// score (highest first) and truncated to `top_k`; a box is then suppressed when
/// its IoU with an already-kept box exceeds `iou_threshold`.
///
/// Returns the kept indices into `boxes`, highest score first.
std::vector<int> nms(const std::vector<cv::Rect2f>& boxes, const std::vector<float>& scores,
                     float iou_threshold, float score_threshold = 0.0f, int top_k = 300);

/// NMS run independently per class id: boxes with different class ids never
/// suppress one another. Same return contract as nms().
std::vector<int> nmsClassAware(const std::vector<cv::Rect2f>& boxes,
                               const std::vector<float>& scores,
                               const std::vector<int>& class_ids, float iou_threshold,
                               float score_threshold = 0.0f, int top_k = 300);

}  // namespace s3m

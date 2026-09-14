#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "s3m/core/types.hpp"

namespace s3m {

/// Classical HSV color-histogram appearance descriptor -- not a learned ReID
/// embedding. That's a deliberate scope choice, not an oversight: it needs no
/// model file, export step, or new dependency, and histogram-based appearance
/// matching is a legitimate, long-established cue (predates deep ReID by
/// decades) for exactly the role it plays here -- one fusable signal among
/// several, not the sole association decision a real ReID network is
/// optimised to be. See docs/roadmap.md Phase 5 for the reasoning.
///
/// Computed over the box's H/S channels only (V/brightness excluded for
/// crude illumination robustness), L1-normalised so compareHist's
/// Bhattacharyya distance is well-defined. Returns an empty Mat if the box
/// doesn't overlap the image at all.
cv::Mat computeAppearanceDescriptor(const cv::Mat& image_bgr, const cv::Rect2f& box);

/// Bhattacharyya distance between two descriptors, in [0, 1] (0 = identical
/// distribution, 1 = disjoint support). Precondition: both non-empty, as
/// returned by computeAppearanceDescriptor for a box that overlapped the
/// image -- callers with a possibly-missing descriptor must branch before
/// calling this rather than pass an empty Mat in.
double appearanceDistance(const cv::Mat& a, const cv::Mat& b);

/// Fills in `.appearance` on every detection from its box's crop in
/// `image_bgr` (best-effort: a box outside the image just leaves it empty,
/// not an error).
void attachAppearance(std::vector<Detection3D>& detections, const cv::Mat& image_bgr);

}  // namespace s3m

#pragma once

#include <opencv2/core.hpp>

namespace s3m {

/// Aspect-preserving resize + centre-pad to a square, the pre-processing used by
/// the YOLO family. Records the transform so detections in the letterboxed frame
/// can be mapped back to the original image.
struct Letterbox {
    float scale = 1.0f;         ///< original_px * scale  ->  letterboxed_px
    int pad_x = 0;              ///< left padding added after scaling
    int pad_y = 0;              ///< top padding
    cv::Size scaled{0, 0};      ///< size of the scaled (pre-pad) image
    cv::Size target{0, 0};      ///< final square size

    /// Map a box from letterboxed pixel space back to the original image.
    cv::Rect2f toOriginal(const cv::Rect2f& box) const;
    /// Map a point from letterboxed pixel space back to the original image.
    cv::Point2f toOriginal(const cv::Point2f& p) const;
};

/// Resize `src` into `dst` (`target` x `target`, same type as `src`), letterboxed
/// with `pad_value`. Returns the transform used.
Letterbox makeLetterbox(const cv::Mat& src, cv::Mat& dst, int target,
                        const cv::Scalar& pad_value = cv::Scalar::all(114));

}  // namespace s3m

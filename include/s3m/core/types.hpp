#pragma once

#include <cstdint>

#include <opencv2/core.hpp>

namespace s3m {

/// A rectified stereo image pair, optionally carrying ground-truth depth/disparity.
///
/// Convention: the left camera is the reference frame. Camera axes are
/// X-right, Y-down, Z-forward (OpenCV convention). Disparity is expressed in
/// pixels and is positive for points in front of the camera.
struct StereoFrame {
    std::int64_t index = 0;  ///< frame number within the sequence
    double timestamp = 0.0;  ///< seconds

    cv::Mat left;   ///< rectified left image  (8U, 1 or 3 channels)
    cv::Mat right;  ///< rectified right image (8U, 1 or 3 channels)

    cv::Mat gt_disparity;  ///< optional CV_32F, pixels; <= 0 or NaN means "no data"
    cv::Mat gt_depth;      ///< optional CV_32F, metres; <= 0 or NaN means "no data"

    bool hasGtDisparity() const { return !gt_disparity.empty(); }
    bool hasGtDepth() const { return !gt_depth.empty(); }
    bool empty() const { return left.empty() || right.empty(); }
    cv::Size size() const { return left.size(); }
};

/// Axis-aligned 2D detection in the left image.
struct Detection2D {
    cv::Rect2f box;      ///< pixel coordinates in the left image
    float score = 0.0f;  ///< detector confidence in [0, 1]
    int class_id = 0;
};

/// A 2D detection promoted to 3D using the depth map.
struct Detection3D {
    cv::Rect2f box;             ///< source 2D box (left image)
    cv::Point3f position{};     ///< metres, left-camera frame
    float depth = 0.0f;         ///< metres (== position.z); robust estimate over the box
    float score = 0.0f;
    int class_id = 0;
    bool valid = false;         ///< false when the box had too few valid depth pixels
};

/// Filtered state of one tracked object.
struct TrackState {
    int id = -1;
    cv::Point3f position{};      ///< filtered position [m], left-camera frame
    cv::Point3f velocity{};      ///< filtered velocity [m/s]
    int age = 0;                 ///< frames since the track was created
    int hits = 0;                ///< total matched detections
    int time_since_update = 0;   ///< frames since the last matched detection
    bool confirmed = false;      ///< promoted past the min-hits threshold
};

}  // namespace s3m

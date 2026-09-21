#pragma once

#include <opencv2/core.hpp>

#include "s3m/camera/camera_model.hpp"

namespace s3m {

/// A calibrated, rectified horizontal stereo pair.
///
/// Geometry (left camera is the reference frame, right camera centre at
/// +baseline along the left X axis):
///
///     d = fx * baseline / Z - doffs           with doffs = cx_right - cx_left
///     Z = fx * baseline / (d + doffs)
///
/// `doffs` is 0 for a perfectly rectified pair and non-zero for datasets such
/// as Middlebury 2014 that keep a principal-point offset between the views.
class StereoRig {
 public:
    StereoRig() = default;
    StereoRig(CameraModel left, CameraModel right, double baseline);

    /// Identical-intrinsics helper (doffs = 0).
    static StereoRig fromIntrinsics(double fx, double fy, double cx, double cy, cv::Size image_size,
                                    double baseline);

    const CameraModel& left() const { return left_; }
    const CameraModel& right() const { return right_; }
    double baseline() const { return baseline_; }  ///< metres, > 0
    double doffs() const { return doffs_; }        ///< cx_right - cx_left, pixels
    void setDoffs(double doffs) { doffs_ = doffs; }

    /// Z [m] for a disparity [px]. Returns +infinity when (d + doffs) <= 0.
    double disparityToDepth(double disparity) const;

    /// Disparity [px] for a depth Z [m].
    double depthToDisparity(double depth) const;

    /// 3D point (left-camera frame) for a left-image pixel and its disparity.
    cv::Point3d triangulate(const cv::Point2d& px_left, double disparity) const;

    /// 4x4 reprojection matrix Q with  [X Y Z W]^T = Q * [u v d 1]^T,
    /// compatible with cv::reprojectImageTo3D applied to the left image.
    cv::Matx44d reprojectionMatrix() const;

    bool valid() const { return left_.valid() && right_.valid() && baseline_ > 0.0; }

 private:
    CameraModel left_;
    CameraModel right_;
    double baseline_ = 0.0;
    double doffs_ = 0.0;
};

}  // namespace s3m

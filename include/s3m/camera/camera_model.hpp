#pragma once

#include <opencv2/core.hpp>

namespace s3m {

/// Pinhole camera intrinsics with optional radial-tangential distortion.
///
/// Camera frame follows the OpenCV convention: X right, Y down, Z forward.
/// project()/backProject() use the ideal pinhole model (distortion is stored
/// for rectification/undistortion helpers but not applied by these two).
class CameraModel {
 public:
    CameraModel() = default;
    CameraModel(double fx, double fy, double cx, double cy, cv::Size image_size,
                cv::Mat dist_coeffs = cv::Mat());

    /// Build from a 3x3 intrinsic matrix K.
    static CameraModel fromMatrix(const cv::Matx33d& K, cv::Size image_size,
                                  cv::Mat dist_coeffs = cv::Mat());

    double fx() const { return fx_; }
    double fy() const { return fy_; }
    double cx() const { return cx_; }
    double cy() const { return cy_; }
    cv::Size imageSize() const { return image_size_; }

    cv::Matx33d K() const;
    const cv::Mat& distCoeffs() const { return dist_coeffs_; }
    bool hasDistortion() const;

    /// Project a camera-frame point to pixel coordinates. Requires Xc.z > 0.
    cv::Point2d project(const cv::Point3d& Xc) const;

    /// Back-project a pixel to a camera-frame point at depth Z (metres).
    cv::Point3d backProject(const cv::Point2d& px, double depth) const;

    /// Unit-length ray direction (camera frame) through a pixel.
    cv::Point3d ray(const cv::Point2d& px) const;

    /// Intrinsics for the same camera after resizing the image by (sx, sy).
    CameraModel scaled(double sx, double sy) const;

    bool valid() const { return fx_ > 0.0 && fy_ > 0.0; }

 private:
    double fx_ = 0.0;
    double fy_ = 0.0;
    double cx_ = 0.0;
    double cy_ = 0.0;
    cv::Size image_size_{0, 0};
    cv::Mat dist_coeffs_;  ///< 1xN CV_64F: [k1, k2, p1, p2, (k3, ...)]
};

}  // namespace s3m

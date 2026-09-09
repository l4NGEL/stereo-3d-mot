#include "s3m/camera/stereo_rig.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace s3m {

StereoRig::StereoRig(CameraModel left, CameraModel right, double baseline)
    : left_(std::move(left)), right_(std::move(right)), baseline_(baseline) {
    if (baseline_ <= 0.0) {
        throw std::invalid_argument("StereoRig: baseline must be > 0");
    }
    doffs_ = right_.cx() - left_.cx();
}

StereoRig StereoRig::fromIntrinsics(double fx, double fy, double cx, double cy,
                                    cv::Size image_size, double baseline) {
    const CameraModel camera(fx, fy, cx, cy, image_size);
    return StereoRig(camera, camera, baseline);
}

double StereoRig::disparityToDepth(double disparity) const {
    const double denominator = disparity + doffs_;
    if (denominator <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    return left_.fx() * baseline_ / denominator;
}

double StereoRig::depthToDisparity(double depth) const {
    if (depth <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    return left_.fx() * baseline_ / depth - doffs_;
}

cv::Point3d StereoRig::triangulate(const cv::Point2d& px_left, double disparity) const {
    const double z = disparityToDepth(disparity);
    if (!std::isfinite(z)) {
        return {0.0, 0.0, std::numeric_limits<double>::infinity()};
    }
    return left_.backProject(px_left, z);
}

cv::Matx44d StereoRig::reprojectionMatrix() const {
    const double f = left_.fx();
    const double cx = left_.cx();
    const double cy = left_.cy();
    const double b = baseline_;
    // [X Y Z W]^T = Q * [u v d 1]^T,  with  W = (d + doffs) / b  and  Z/W = f*b/(d + doffs).
    return cv::Matx44d(1.0, 0.0, 0.0,      -cx,
                       0.0, 1.0, 0.0,      -cy,
                       0.0, 0.0, 0.0,        f,
                       0.0, 0.0, 1.0 / b, doffs_ / b);
}

}  // namespace s3m

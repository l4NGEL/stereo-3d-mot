#include "s3m/camera/camera_model.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace s3m {

CameraModel::CameraModel(double fx, double fy, double cx, double cy, cv::Size image_size,
                         cv::Mat dist_coeffs)
    : fx_(fx), fy_(fy), cx_(cx), cy_(cy), image_size_(image_size) {
    if (!dist_coeffs.empty()) {
        cv::Mat as_double;
        dist_coeffs.convertTo(as_double, CV_64F);
        dist_coeffs_ = as_double.reshape(1, 1).clone();
    }
}

CameraModel CameraModel::fromMatrix(const cv::Matx33d& K, cv::Size image_size,
                                    cv::Mat dist_coeffs) {
    return CameraModel(K(0, 0), K(1, 1), K(0, 2), K(1, 2), image_size, std::move(dist_coeffs));
}

cv::Matx33d CameraModel::K() const {
    return cv::Matx33d(fx_, 0.0, cx_, 0.0, fy_, cy_, 0.0, 0.0, 1.0);
}

bool CameraModel::hasDistortion() const {
    return !dist_coeffs_.empty() && cv::countNonZero(dist_coeffs_) > 0;
}

cv::Point2d CameraModel::project(const cv::Point3d& Xc) const {
    if (Xc.z <= 0.0) {
        throw std::invalid_argument("CameraModel::project: point is not in front of the camera");
    }
    return {fx_ * Xc.x / Xc.z + cx_, fy_ * Xc.y / Xc.z + cy_};
}

cv::Point3d CameraModel::backProject(const cv::Point2d& px, double depth) const {
    return {(px.x - cx_) * depth / fx_, (px.y - cy_) * depth / fy_, depth};
}

cv::Point3d CameraModel::ray(const cv::Point2d& px) const {
    cv::Point3d dir((px.x - cx_) / fx_, (px.y - cy_) / fy_, 1.0);
    const double norm = std::sqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
    return {dir.x / norm, dir.y / norm, dir.z / norm};
}

CameraModel CameraModel::scaled(double sx, double sy) const {
    const cv::Size resized(cvRound(image_size_.width * sx), cvRound(image_size_.height * sy));
    return CameraModel(fx_ * sx, fy_ * sy, cx_ * sx, cy_ * sy, resized, dist_coeffs_.clone());
}

}  // namespace s3m

#include "s3m/geometry/reprojection.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "s3m/depth/stereo_matcher.hpp"

namespace s3m {
namespace {

inline bool isValidDisparity(float d) {
    return std::isfinite(d) && d != StereoMatcher::kInvalidDisparity;
}

}  // namespace

cv::Mat reprojectWithQ(const cv::Mat& disparity, const cv::Matx44d& Q, cv::Mat* valid_mask) {
    CV_Assert(disparity.type() == CV_32F);

    cv::Mat points(disparity.size(), CV_32FC3, cv::Scalar::all(0));
    cv::Mat mask(disparity.size(), CV_8U, cv::Scalar::all(0));

    for (int y = 0; y < disparity.rows; ++y) {
        const float* d_row = disparity.ptr<float>(y);
        cv::Vec3f* p_row = points.ptr<cv::Vec3f>(y);
        uchar* m_row = mask.ptr<uchar>(y);
        for (int x = 0; x < disparity.cols; ++x) {
            if (!isValidDisparity(d_row[x]))
                continue;

            const double u = static_cast<double>(x);
            const double v = static_cast<double>(y);
            const double d = static_cast<double>(d_row[x]);

            const double X = Q(0, 0) * u + Q(0, 1) * v + Q(0, 2) * d + Q(0, 3);
            const double Y = Q(1, 0) * u + Q(1, 1) * v + Q(1, 2) * d + Q(1, 3);
            const double Z = Q(2, 0) * u + Q(2, 1) * v + Q(2, 2) * d + Q(2, 3);
            const double W = Q(3, 0) * u + Q(3, 1) * v + Q(3, 2) * d + Q(3, 3);
            if (std::abs(W) < 1e-9)
                continue;

            const double z = Z / W;
            if (!std::isfinite(z) || z <= 0.0)
                continue;

            p_row[x] = cv::Vec3f(static_cast<float>(X / W), static_cast<float>(Y / W),
                                 static_cast<float>(z));
            m_row[x] = 255;
        }
    }

    if (valid_mask)
        *valid_mask = mask;
    return points;
}

cv::Mat reproject(const cv::Mat& disparity, const StereoRig& rig, cv::Mat* valid_mask) {
    return reprojectWithQ(disparity, rig.reprojectionMatrix(), valid_mask);
}

cv::Mat disparityToDepthMap(const cv::Mat& disparity, const StereoRig& rig) {
    CV_Assert(disparity.type() == CV_32F);
    const double f_baseline = rig.left().fx() * rig.baseline();
    const double doffs = rig.doffs();

    cv::Mat depth(disparity.size(), CV_32F, cv::Scalar::all(0));
    for (int y = 0; y < disparity.rows; ++y) {
        const float* d_row = disparity.ptr<float>(y);
        float* z_row = depth.ptr<float>(y);
        for (int x = 0; x < disparity.cols; ++x) {
            if (!isValidDisparity(d_row[x]))
                continue;
            const double denominator = static_cast<double>(d_row[x]) + doffs;
            if (denominator <= 0.0)
                continue;
            z_row[x] = static_cast<float>(f_baseline / denominator);
        }
    }
    return depth;
}

float robustDepthInRoi(const cv::Mat& depth_map, const cv::Rect& roi, float shrink,
                       float min_valid_fraction) {
    CV_Assert(depth_map.type() == CV_32F);
    const cv::Rect image_rect(0, 0, depth_map.cols, depth_map.rows);

    const float s = std::clamp(shrink, 0.0f, 0.45f);
    const int dx = static_cast<int>(std::lround(static_cast<float>(roi.width) * s));
    const int dy = static_cast<int>(std::lround(static_cast<float>(roi.height) * s));

    cv::Rect region(roi.x + dx, roi.y + dy, roi.width - 2 * dx, roi.height - 2 * dy);
    region &= image_rect;
    if (region.width <= 0 || region.height <= 0)
        return -1.0f;

    std::vector<float> values;
    values.reserve(static_cast<std::size_t>(region.area()));
    for (int y = region.y; y < region.y + region.height; ++y) {
        const float* row = depth_map.ptr<float>(y);
        for (int x = region.x; x < region.x + region.width; ++x) {
            if (std::isfinite(row[x]) && row[x] > 0.0f)
                values.push_back(row[x]);
        }
    }

    const float fraction =
        static_cast<float>(values.size()) / static_cast<float>(std::max(1, region.area()));
    if (values.empty() || fraction < min_valid_fraction)
        return -1.0f;

    const std::size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid),
                     values.end());
    return values[mid];
}

std::vector<Detection3D> promoteTo3D(const std::vector<Detection2D>& detections,
                                     const cv::Mat& depth_map, const StereoRig& rig) {
    std::vector<Detection3D> result;
    result.reserve(detections.size());
    const cv::Rect image_rect(0, 0, depth_map.cols, depth_map.rows);

    for (const Detection2D& det : detections) {
        Detection3D d3;
        d3.box = det.box;
        d3.score = det.score;
        d3.class_id = det.class_id;

        const cv::Rect roi = cv::Rect(cvRound(det.box.x), cvRound(det.box.y),
                                      cvRound(det.box.width), cvRound(det.box.height)) &
                             image_rect;
        const float z =
            (roi.width > 0 && roi.height > 0) ? robustDepthInRoi(depth_map, roi) : -1.0f;

        if (z > 0.0f) {
            const cv::Point2d center(static_cast<double>(det.box.x) + det.box.width / 2.0,
                                     static_cast<double>(det.box.y) + det.box.height / 2.0);
            const cv::Point3d p = rig.left().backProject(center, static_cast<double>(z));
            d3.position = cv::Point3f(static_cast<float>(p.x), static_cast<float>(p.y),
                                      static_cast<float>(p.z));
            d3.depth = z;
            d3.valid = true;
        }
        result.push_back(d3);
    }
    return result;
}

}  // namespace s3m

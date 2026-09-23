#include "s3m/vo/visual_odometry.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace s3m {
namespace {

cv::Mat toGray(const cv::Mat& image) {
    if (image.channels() == 1) return image;
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

cv::Mat toCvMat(const cv::Matx33d& m) { return cv::Mat(m); }

}  // namespace

VisualOdometry::VisualOdometry(const StereoRig& rig, VisualOdometryParams params)
    : rig_(rig),
      params_(params),
      orb_(cv::ORB::create(params.max_features)),
      matcher_(cv::BFMatcher::create(cv::NORM_HAMMING)),
      stereo_matcher_(params.stereo_matcher) {}

void VisualOdometry::reset() {
    has_prev_ = false;
    prev_ = FrameFeatures{};
    R_wc_ = cv::Matx33d::eye();
    C_world_ = cv::Point3d();
}

VisualOdometry::FrameFeatures VisualOdometry::extract(const cv::Mat& left, const cv::Mat& right) {
    FrameFeatures f;
    const cv::Mat left_gray = toGray(left);
    orb_->detectAndCompute(left_gray, cv::noArray(), f.keypoints, f.descriptors);

    const cv::Mat disparity = stereo_matcher_.computeDisparity(left, right);
    f.points3d.resize(f.keypoints.size());
    f.valid3d.assign(f.keypoints.size(), 0);
    for (std::size_t i = 0; i < f.keypoints.size(); ++i) {
        const cv::Point2f& pt = f.keypoints[i].pt;
        const int x = cvRound(pt.x);
        const int y = cvRound(pt.y);
        if (x < 0 || x >= disparity.cols || y < 0 || y >= disparity.rows) continue;
        const float d = disparity.at<float>(y, x);
        if (!std::isfinite(d) || d == StereoMatcher::kInvalidDisparity) continue;
        f.points3d[i] = rig_.triangulate(cv::Point2d(pt.x, pt.y), d);
        f.valid3d[i] = 1;
    }
    return f;
}

VoFrameResult VisualOdometry::processFrame(const cv::Mat& left, const cv::Mat& right) {
    FrameFeatures curr = extract(left, right);

    VoFrameResult result;
    if (!has_prev_) {
        R_wc_ = cv::Matx33d::eye();
        C_world_ = cv::Point3d();
        result.ok = true;
        result.pose.rotation = R_wc_.t();  // camera->world = (world->camera)^-1 = transpose (orthonormal)
        result.pose.position = C_world_;
        prev_ = std::move(curr);
        has_prev_ = true;
        return result;
    }

    // Coast on failure: report the held pose so the trajectory stays
    // continuous, but still hand the tracker this frame's features to try
    // matching against next time -- a single bad frame shouldn't strand
    // tracking on a stale keyframe indefinitely.
    const auto coast = [&]() {
        result.ok = false;
        result.pose.rotation = R_wc_.t();
        result.pose.position = C_world_;
        prev_ = std::move(curr);
        return result;
    };

    if (prev_.descriptors.empty() || curr.descriptors.empty()) return coast();

    std::vector<std::vector<cv::DMatch>> knn;
    matcher_->knnMatch(prev_.descriptors, curr.descriptors, knn, 2);
    std::vector<cv::DMatch> good;
    good.reserve(knn.size());
    for (const auto& m : knn) {
        if (m.size() == 2 && m[0].distance < params_.match_ratio * m[1].distance) good.push_back(m[0]);
    }
    result.num_matches = static_cast<int>(good.size());
    if (result.num_matches < params_.min_inliers) return coast();

    std::vector<cv::Point2f> pts_prev(good.size());
    std::vector<cv::Point2f> pts_curr(good.size());
    for (std::size_t k = 0; k < good.size(); ++k) {
        pts_prev[k] = prev_.keypoints[static_cast<std::size_t>(good[k].queryIdx)].pt;
        pts_curr[k] = curr.keypoints[static_cast<std::size_t>(good[k].trainIdx)].pt;
    }

    const cv::Mat K = toCvMat(rig_.left().K());
    cv::Mat inlier_mask;
    const cv::Mat E = cv::findEssentialMat(pts_prev, pts_curr, K, cv::RANSAC, params_.ransac_prob,
                                           params_.ransac_threshold, inlier_mask);
    if (E.empty() || E.rows != 3 || E.cols != 3) return coast();

    cv::Mat R_cv;
    cv::Mat t_cv;
    const int inliers = cv::recoverPose(E, pts_prev, pts_curr, K, R_cv, t_cv, inlier_mask);
    result.num_inliers = inliers;
    if (inliers < params_.min_inliers) return coast();

    cv::Matx33d R_rel;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) R_rel(r, c) = R_cv.at<double>(r, c);
    const cv::Vec3d t_hat(t_cv.at<double>(0), t_cv.at<double>(1), t_cv.at<double>(2));

    // Scale from stereo: median over inlier matches that also have a valid
    // stereo-triangulated 3D point on both sides -- see the class comment.
    std::vector<double> scale_candidates;
    scale_candidates.reserve(good.size());
    for (std::size_t k = 0; k < good.size(); ++k) {
        if (!inlier_mask.at<uchar>(static_cast<int>(k))) continue;
        const std::size_t qi = static_cast<std::size_t>(good[k].queryIdx);
        const std::size_t ti = static_cast<std::size_t>(good[k].trainIdx);
        if (!prev_.valid3d[qi] || !curr.valid3d[ti]) continue;
        const cv::Vec3d Xp(prev_.points3d[qi].x, prev_.points3d[qi].y, prev_.points3d[qi].z);
        const cv::Vec3d Xc(curr.points3d[ti].x, curr.points3d[ti].y, curr.points3d[ti].z);
        const cv::Vec3d diff = Xc - (R_rel * Xp);
        scale_candidates.push_back(diff.dot(t_hat));
    }
    if (scale_candidates.size() < 3) return coast();  // too few to trust a median
    std::sort(scale_candidates.begin(), scale_candidates.end());
    const double scale = scale_candidates[scale_candidates.size() / 2];
    if (!std::isfinite(scale) || scale <= 0.0) return coast();  // moving backward-only is not a VO bug worth trusting blindly

    const cv::Vec3d t_scaled = scale * t_hat;

    // Global pose composition (validated): R_wc_new = R_rel * R_wc_old;
    // C_new = C_old - R_wc_new^T * t_scaled.
    const cv::Matx33d R_wc_new = R_rel * R_wc_;
    const cv::Vec3d C_old(C_world_.x, C_world_.y, C_world_.z);
    const cv::Vec3d C_new = C_old - R_wc_new.t() * t_scaled;

    R_wc_ = R_wc_new;
    C_world_ = cv::Point3d(C_new[0], C_new[1], C_new[2]);

    result.ok = true;
    result.scale = scale;
    result.pose.rotation = R_wc_.t();
    result.pose.position = C_world_;
    prev_ = std::move(curr);
    return result;
}

}  // namespace s3m

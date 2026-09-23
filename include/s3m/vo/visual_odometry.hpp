#pragma once

#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "s3m/camera/stereo_rig.hpp"
#include "s3m/depth/stereo_matcher.hpp"
#include "s3m/io/kitti_odometry_loader.hpp"

namespace s3m {

struct VisualOdometryParams {
    int max_features = 2000;        ///< ORB
    float match_ratio = 0.75f;      ///< Lowe's ratio test on the top-2 kNN match
    double ransac_prob = 0.999;     ///< findEssentialMat
    double ransac_threshold = 1.0;  ///< findEssentialMat, px
    /// Below this many ratio-test-passing matches, or below this many
    /// cheirality-inlier matches from recoverPose, the frame is untrusted:
    /// the pose from the last *trusted* frame is held (coasted) rather than
    /// updated from a geometrically underdetermined estimate. 5 is the
    /// essential matrix's own hard minimum; the default is well above it so
    /// "barely enough points" doesn't get treated as "trustworthy".
    int min_inliers = 15;
    StereoMatcherParams stereo_matcher;  ///< for per-frame stereo scale recovery
};

/// One frame's VO output. `ok == false` means tracking was untrusted this
/// frame (too few matches/inliers, or a degenerate scale estimate) and the
/// pose was coasted from the last trusted frame rather than updated -- still
/// usable for a continuous trajectory, just not a fresh measurement.
struct VoFrameResult {
    bool ok = false;
    Pose pose;             ///< camera-to-world; see kitti_odometry_loader.hpp
    int num_matches = 0;   ///< ratio-test-passing matches against the previous frame
    int num_inliers = 0;   ///< of those, recoverPose's cheirality-check inliers
    double scale = 0.0;    ///< recovered translation scale (diagnostic; 0 if not ok)
};

/// Frame-to-frame stereo visual odometry: ORB features matched between
/// consecutive LEFT frames, relative rotation + translation *direction* from
/// the 5-point essential matrix (cv::findEssentialMat / cv::recoverPose),
/// translation *scale* resolved from this project's own stereo depth (the
/// classic monocular-VO scale-ambiguity problem, solved with the stereo rig
/// already built in Phases 1-6 rather than left unresolved or guessed).
///
/// Scale recovery: for each essential-matrix inlier match with a valid
/// stereo-triangulated 3D point on both sides (X_prev, X_curr, both in their
/// own frame's camera coordinates), the relation
/// `X_curr = R_rel * X_prev + s * t_hat` (R_rel, t_hat from recoverPose, s
/// the unknown true scale, t_hat unit-norm) is solved per-point for
/// `s_i = dot(X_curr_i - R_rel * X_prev_i, t_hat)`, and the median across all
/// such points is taken as the frame's scale -- robust to a few bad
/// triangulations without needing a full outlier-rejection pass of its own.
/// Validated against a known synthetic camera motion before porting to C++
/// (see the Phase 7 write-up in docs/roadmap.md); recovers ground-truth
/// rotation, scale and position to floating-point precision there.
class VisualOdometry {
 public:
    explicit VisualOdometry(const StereoRig& rig, VisualOdometryParams params = {});

    /// Process one rectified stereo pair. The first call always returns
    /// `ok = true` with the identity pose (this frame becomes the trajectory
    /// origin, matching the KITTI odometry ground truth's own convention).
    VoFrameResult processFrame(const cv::Mat& left, const cv::Mat& right);

    void reset();

 private:
    struct FrameFeatures {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        std::vector<cv::Point3d> points3d;
        std::vector<char> valid3d;
    };

    FrameFeatures extract(const cv::Mat& left, const cv::Mat& right);

    StereoRig rig_;
    VisualOdometryParams params_;
    cv::Ptr<cv::ORB> orb_;
    cv::Ptr<cv::BFMatcher> matcher_;
    StereoMatcher stereo_matcher_;

    bool has_prev_ = false;
    FrameFeatures prev_;

    cv::Matx33d R_wc_ = cv::Matx33d::eye();  ///< current world -> camera rotation
    cv::Point3d C_world_;                     ///< current camera center, world frame
};

}  // namespace s3m

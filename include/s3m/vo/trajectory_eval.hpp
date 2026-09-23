#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "s3m/io/kitti_odometry_loader.hpp"

namespace s3m {

struct RigidAlignment {
    cv::Matx33d rotation = cv::Matx33d::eye();
    cv::Point3d translation;
};

/// Kabsch/Umeyama rigid (rotation + translation, no scale -- this project's
/// VO already recovers metric scale from stereo, so aligning with a free
/// scale factor would hide a real scale error rather than measure it)
/// alignment minimising sum ||target_i - (R*source_i + t)||^2. Throws
/// std::invalid_argument if the two point sets differ in size or have fewer
/// than 3 points.
RigidAlignment alignRigid(const std::vector<cv::Point3d>& source, const std::vector<cv::Point3d>& target);

struct TrajectoryError {
    double ate_rmse = 0.0;        ///< metres, after optimal SE(3) alignment to ground truth
    double ate_mean = 0.0;        ///< metres
    double rpe_trans_rmse = 0.0;  ///< metres, per `delta`-frame relative step
    double rpe_rot_rmse_deg = 0.0;
    int num_poses = 0;
    int rpe_num_pairs = 0;
};

/// ATE (Absolute Trajectory Error, RMSE of per-frame position error after
/// Kabsch/Umeyama-aligning the estimated trajectory onto ground truth) and
/// RPE (Relative Pose Error over `delta`-frame steps) -- the standard
/// definitions from Sturm et al. 2012 (the TUM RGB-D benchmark's own
/// metrics; the KITTI odometry devkit's own evaluation is a close relative).
/// `estimated` and `ground_truth` must be the same length and frame-aligned
/// (estimated[i] and ground_truth[i] are the same instant).
TrajectoryError evaluateTrajectory(const std::vector<Pose>& estimated,
                                   const std::vector<Pose>& ground_truth, int delta = 1);

}  // namespace s3m

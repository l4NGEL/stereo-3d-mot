#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "s3m/io/frame_source.hpp"
#include "s3m/io/kitti_loader.hpp"

namespace s3m {

/// A camera-to-world rigid pose: `Xworld = rotation * Xcamera + position`, so
/// `position` is exactly the camera's center in the world frame (setting
/// Xcamera = 0). Matches the KITTI odometry benchmark's own convention (world
/// = frame 0's own camera frame) so ground truth and estimated trajectories
/// compare directly with no extra transpose/inverse at the comparison site.
struct Pose {
    cv::Matx33d rotation = cv::Matx33d::eye();  ///< camera -> world
    cv::Point3d position;                       ///< camera center, world frame
};

/// Parses one poses/<seq>.txt (KITTI odometry devkit format): one line per
/// frame, 12 whitespace-separated numbers = a row-major flattened 3x4
/// [R | t] camera-to-world matrix. Throws std::runtime_error if the file
/// cannot be opened, or a non-empty line doesn't parse to exactly 12 numbers.
std::vector<Pose> readKittiPoses(const std::string& path);

/// Same calib.txt format as the tracking benchmark (see readKittiCalib), but
/// the odometry benchmark's grayscale stereo pair is P0 (left, "image_0") /
/// P1 (right, "image_1"), not P2/P3.
StereoRig readKittiOdometryCalib(const std::string& path, cv::Size image_size);

/// One KITTI *odometry* sequence -- a different benchmark from
/// KittiTrackingSource: no object labels, but (for sequences 00-10) real
/// ego-motion ground truth, which is what Phase 7 (visual odometry) needs to
/// evaluate against. Expected layout under `root` (see
/// scripts/fetch_kitti_odometry.py):
///   sequences/<sequence>/calib.txt
///   sequences/<sequence>/image_0/000000.png, ...   (left, grayscale)
///   sequences/<sequence>/image_1/000000.png, ...   (right, grayscale)
///   poses/<sequence>.txt                            (absent for the
///                                                    benchmark's own held-out
///                                                    test sequences, 11-21)
class KittiOdometrySource : public FrameSource {
 public:
    KittiOdometrySource(const std::string& root, const std::string& sequence);

    const StereoRig& rig() const override { return rig_; }
    int size() const override { return frame_count_; }
    std::optional<StereoFrame> next() override;
    void reset() override { cursor_ = 0; }

    bool hasGroundTruth() const { return !poses_.empty(); }
    /// Empty if hasGroundTruth() is false. Parallel to frame index (poses[i]
    /// is the pose at the i-th call to next()).
    const std::vector<Pose>& groundTruthPoses() const { return poses_; }

 private:
    StereoRig rig_;
    std::string image0_dir_;
    std::string image1_dir_;
    int frame_count_ = 0;
    std::vector<Pose> poses_;
    int cursor_ = 0;
};

}  // namespace s3m

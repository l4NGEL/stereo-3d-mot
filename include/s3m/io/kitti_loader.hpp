#pragma once

#include <map>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "s3m/io/frame_source.hpp"

namespace s3m {

/// One row of a KITTI tracking-benchmark label file (label_02/<seq>.txt), or a
/// results file in the same format (score present). Field names and units
/// match the KITTI devkit exactly: `bbox` is left-image pixels; `location` is
/// the *bottom-center* of the 3D box (not the centroid) in the rectified
/// camera-2 frame, metres; `rotation_y` is yaw around the camera Y axis.
struct KittiObject {
    int frame = 0;
    int track_id = -1;   ///< -1 marks a DontCare region, not a real track
    std::string type;    ///< "Car", "Pedestrian", "Cyclist", "DontCare", ...
    double truncated = 0.0;
    int occluded = 0;    ///< 0 visible, 1 partly, 2 largely, 3 unknown
    double alpha = 0.0;
    cv::Rect2f bbox;
    double height = 0.0;
    double width = 0.0;
    double length = 0.0;
    cv::Point3d location;   ///< bottom-center, camera-2 rectified frame [m]
    double rotation_y = 0.0;
    double score = -1.0;    ///< only in result files; -1 means "not present" (GT)

    bool isDontCare() const { return type == "DontCare"; }
    /// Bottom-center + half height: a point roughly at the object's 3D middle,
    /// a closer match to what a depth-based detector like this project's
    /// promoteTo3D() would report than the raw bottom-center location.
    cv::Point3d centroid() const { return {location.x, location.y - height / 2.0, location.z}; }
};

/// Parse one label_02/<seq>.txt (ground truth, 17 whitespace-separated fields
/// per line) or a results file in the same format (18 fields, trailing
/// score). Malformed individual lines are skipped, not fatal. Throws
/// std::runtime_error if the file cannot be opened.
std::vector<KittiObject> readKittiLabels(const std::string& path);

/// Parse one calib/<seq>.txt (KITTI tracking-benchmark calibration: `P0`..`P3`
/// 3x4 projection matrices, `R_rect`, `Tr_velo_cam`, `Tr_imu_velo`) and build a
/// StereoRig from the colour stereo pair P2 (left, "image_02") / P3 (right,
/// "image_03") -- the images the tracking benchmark ships. For a rectified
/// projection matrix P = [K | K*t], the left 3x3 block *is* K, and the
/// per-camera offset along X is `-P[0,3]/K[0,0]`; the baseline used here is
/// the difference between cameras 2 and 3. Throws std::runtime_error if the
/// file is missing P2 or P3.
StereoRig readKittiCalib(const std::string& path, cv::Size image_size);

/// One KITTI tracking sequence. Expected layout under `root`:
///   calib/<sequence>.txt
///   image_02/<sequence>/000000.png, 000001.png, ...   (left, colour)
///   image_03/<sequence>/000000.png, ...                (right, colour)
///   label_02/<sequence>.txt                             (optional: absent
///                                                        for the test split)
/// `sequence` is the zero-padded id used in KITTI's own file names, e.g.
/// "0000". A `FrameSource` over the rectified pairs; `objectsAt()` exposes the
/// ground truth (excluding DontCare) for evaluation via MotAccumulator.
class KittiTrackingSource : public FrameSource {
 public:
    KittiTrackingSource(const std::string& root, const std::string& sequence);

    const StereoRig& rig() const override { return rig_; }
    int size() const override { return static_cast<int>(frame_ids_.size()); }
    std::optional<StereoFrame> next() override;
    void reset() override { cursor_ = 0; }

    /// Ground truth for `frame_index` (its position in this sequence, 0-based
    /// -- not necessarily the raw KITTI frame number), excluding DontCare.
    /// Empty if the sequence has no label file (the test split).
    std::vector<KittiObject> objectsAt(int frame_index) const;

    bool hasGroundTruth() const { return !labels_by_frame_.empty(); }

 private:
    StereoRig rig_;
    std::string image02_dir_;
    std::string image03_dir_;
    std::vector<int> frame_ids_;                          ///< KITTI frame numbers, in order
    std::map<int, std::vector<KittiObject>> labels_by_frame_;
    int cursor_ = 0;
};

}  // namespace s3m

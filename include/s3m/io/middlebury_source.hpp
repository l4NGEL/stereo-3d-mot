#pragma once

#include <string>

#include <opencv2/core.hpp>

#include "s3m/io/frame_source.hpp"

namespace s3m {

/// Reader for one Middlebury 2014 stereo scene directory containing:
///   im0.png, im1.png, calib.txt   and optionally   disp0.pfm  (ground truth).
///
/// calib.txt supplies cam0/cam1 (with per-view cx), `doffs`, `baseline` in mm,
/// and the image size. Ground-truth disparity is converted to a depth map with
/// the parsed geometry so both are available on the frame.
class MiddleburySource : public FrameSource {
 public:
    explicit MiddleburySource(const std::string& scene_dir);

    const StereoRig& rig() const override { return rig_; }
    int size() const override { return 1; }
    std::optional<StereoFrame> next() override;
    void reset() override { served_ = false; }

    const std::string& sceneName() const { return scene_name_; }

 private:
    StereoRig rig_;
    std::string scene_name_;
    cv::Mat left_;
    cv::Mat right_;
    cv::Mat gt_disparity_;
    bool served_ = false;
};

}  // namespace s3m

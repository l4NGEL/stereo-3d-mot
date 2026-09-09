#pragma once

#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>

#include "s3m/io/frame_source.hpp"

namespace s3m {

/// Procedurally generated stereo scene with exact ground-truth depth/disparity.
///
/// The scene is a textured background plane plus several fronto-parallel
/// textured "cards" at known depths. The right image is synthesised from the
/// left by depth-dependent horizontal shift (painter's order, near cards last),
/// so the ground-truth disparity is analytically exact up to occlusion borders.
/// Cards drift horizontally between frames to exercise tracking.
class SyntheticStereoSource : public FrameSource {
 public:
    struct Card {
        cv::Rect region;  ///< placement in the left image (frame 0)
        double depth_m;   ///< constant depth [m]
    };

    struct Options {
        cv::Size image_size = cv::Size(640, 480);
        double fx = 500.0;
        double fy = 500.0;
        double cx = -1.0;  ///< <= 0 -> image centre
        double cy = -1.0;
        double baseline_m = 0.12;
        double background_depth_m = 12.0;
        int num_frames = 30;
        std::uint64_t seed = 42;
        double card_drift_px = 1.5;  ///< horizontal drift per frame (left image)
        double noise_std = 3.0;      ///< additive Gaussian sensor noise [gray levels]
        std::vector<Card> cards;     ///< empty -> defaultCards()

        static std::vector<Card> defaultCards();
    };

    SyntheticStereoSource() : SyntheticStereoSource(Options{}) {}
    explicit SyntheticStereoSource(Options options);

    const StereoRig& rig() const override { return rig_; }
    int size() const override { return options_.num_frames; }
    std::optional<StereoFrame> next() override;
    void reset() override { cursor_ = 0; }

    /// Ground-truth left-image boxes for the cards at a given frame.
    std::vector<cv::Rect> cardBoxes(int frame_index) const;

    const Options& options() const { return options_; }

 private:
    StereoFrame render(int frame_index) const;
    cv::Rect cardRegionAt(int card_index, int frame_index) const;

    Options options_;
    StereoRig rig_;
    cv::Mat background_texture_;          ///< oversized horizontally for view synthesis
    std::vector<cv::Mat> card_textures_;  ///< one distinct texture per card
    int cursor_ = 0;
};

}  // namespace s3m

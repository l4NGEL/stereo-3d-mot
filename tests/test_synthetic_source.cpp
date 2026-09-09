#include <gtest/gtest.h>

#include "s3m/depth/depth_metrics.hpp"
#include "s3m/depth/stereo_matcher.hpp"
#include "s3m/geometry/reprojection.hpp"
#include "s3m/io/synthetic_source.hpp"

using namespace s3m;

TEST(SyntheticSource, ProducesFramesWithGroundTruth) {
    SyntheticStereoSource::Options options;
    options.num_frames = 3;
    SyntheticStereoSource source(options);
    EXPECT_EQ(source.size(), 3);

    int count = 0;
    while (const auto frame = source.next()) {
        EXPECT_FALSE(frame->left.empty());
        EXPECT_EQ(frame->left.size(), frame->right.size());
        EXPECT_EQ(frame->left.type(), CV_8UC3);
        ASSERT_TRUE(frame->hasGtDisparity());
        ASSERT_TRUE(frame->hasGtDepth());
        ++count;
    }
    EXPECT_EQ(count, 3);
}

TEST(SyntheticSource, GroundTruthDisparityAndDepthAreConsistent) {
    const SyntheticStereoSource::Options options;
    SyntheticStereoSource source(options);
    const auto frame = source.next();
    ASSERT_TRUE(frame.has_value());

    const StereoRig& rig = source.rig();
    for (int y = 40; y < 240; y += 31) {
        for (int x = 30; x < 620; x += 47) {
            const float d = frame->gt_disparity.at<float>(y, x);
            const float z = frame->gt_depth.at<float>(y, x);
            EXPECT_NEAR(rig.disparityToDepth(d), static_cast<double>(z), 1e-2);
        }
    }
}

TEST(SyntheticSource, CardsDriftBetweenFrames) {
    SyntheticStereoSource::Options options;
    options.num_frames = 10;
    options.card_drift_px = 2.0;
    SyntheticStereoSource source(options);

    const std::vector<cv::Rect> first = source.cardBoxes(0);
    const std::vector<cv::Rect> later = source.cardBoxes(6);
    ASSERT_EQ(first.size(), later.size());
    ASSERT_FALSE(first.empty());
    EXPECT_NE(first[0].x, later[0].x);
}

TEST(SyntheticSource, StereoPipelineRecoversApproximateDepth) {
    SyntheticStereoSource::Options options;
    options.num_frames = 1;
    options.noise_std = 0.0;
    SyntheticStereoSource source(options);
    const auto frame = source.next();
    ASSERT_TRUE(frame.has_value());

    StereoMatcherParams params;
    params.type = "SGBM";
    params.num_disparities = 64;
    params.block_size = 7;
    StereoMatcher matcher(params);

    const cv::Mat disparity = matcher.computeDisparity(frame->left, frame->right);
    const cv::Mat depth = disparityToDepthMap(disparity, source.rig());

    const int crop = 64;
    const cv::Rect roi(crop, 0, depth.cols - crop, depth.rows);
    const DepthMetrics m = evaluate(depth(roi), frame->gt_depth(roi), 0.1, 50.0, {1.0, 2.0});

    // Generous bounds: this guards the geometry / sign conventions end to end,
    // not the matcher's precision. The background plane alone is ~80% of pixels.
    ASSERT_GT(m.gt_valid_pixels, 0);
    EXPECT_GT(m.density, 0.40);
    EXPECT_LT(m.abs_rel, 0.30);
}

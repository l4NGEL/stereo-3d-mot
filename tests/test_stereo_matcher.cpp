#include <stdexcept>

#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>

#include "s3m/depth/stereo_matcher.hpp"

using namespace s3m;

namespace {

// Left: a richly textured image. Right: the same content shifted left by
// `shift` px, so a feature at column c in the left image sits at column c-shift
// in the right image, i.e. the true disparity is `shift`.
void makeShiftedPair(int shift, cv::Mat& left, cv::Mat& right) {
    cv::RNG rng(7);
    left.create(240, 320, CV_8UC1);
    rng.fill(left, cv::RNG::UNIFORM, 0, 256);
    cv::GaussianBlur(left, left, cv::Size(3, 3), 0.0);

    right = cv::Mat::zeros(left.size(), CV_8UC1);
    left(cv::Rect(shift, 0, left.cols - shift, left.rows))
        .copyTo(right(cv::Rect(0, 0, left.cols - shift, left.rows)));
}

}  // namespace

TEST(StereoMatcher, RecoversKnownIntegerShiftSgbm) {
    cv::Mat left;
    cv::Mat right;
    makeShiftedPair(12, left, right);

    StereoMatcherParams params;
    params.type = "SGBM";
    params.num_disparities = 32;
    params.block_size = 5;
    StereoMatcher matcher(params);

    const cv::Mat disparity = matcher.computeDisparity(left, right);
    ASSERT_EQ(disparity.type(), CV_32F);

    const cv::Rect roi(60, 40, 200, 160);
    const cv::Mat disp_roi = disparity(roi);
    const cv::Mat mask = StereoMatcher::validMask(disp_roi);
    ASSERT_GT(cv::countNonZero(mask), roi.area() / 2);

    const cv::Scalar mean = cv::mean(disp_roi, mask);
    EXPECT_NEAR(mean[0], 12.0, 1.5);
}

TEST(StereoMatcher, RecoversKnownIntegerShiftBm) {
    cv::Mat left;
    cv::Mat right;
    makeShiftedPair(16, left, right);

    StereoMatcherParams params;
    params.type = "BM";
    params.num_disparities = 48;
    params.block_size = 15;
    StereoMatcher matcher(params);

    const cv::Mat disparity = matcher.computeDisparity(left, right);
    const cv::Rect roi(90, 50, 160, 140);
    const cv::Mat disp_roi = disparity(roi);
    const cv::Mat mask = StereoMatcher::validMask(disp_roi);
    ASSERT_GT(cv::countNonZero(mask), roi.area() / 4);

    EXPECT_NEAR(cv::mean(disp_roi, mask)[0], 16.0, 2.5);
}

TEST(StereoMatcher, NumDisparitiesRoundedToMultipleOf16) {
    StereoMatcherParams params;
    params.num_disparities = 70;
    StereoMatcher matcher(params);
    EXPECT_EQ(matcher.params().num_disparities % 16, 0);
    EXPECT_GE(matcher.params().num_disparities, 70);
}

TEST(StereoMatcher, UnknownTypeThrows) {
    StereoMatcherParams params;
    params.type = "definitely-not-a-matcher";
    EXPECT_THROW(StereoMatcher{params}, std::invalid_argument);
}

// Phase 6: num_tiles > 1 splits the image into horizontal strips computed in
// parallel (one independent matcher per strip, overlapping margin cropped
// off before stitching -- see stereo_matcher.cpp). Still recovers the same
// known shift as the untiled path.
TEST(StereoMatcher, TiledStillRecoversKnownShift) {
    cv::Mat left;
    cv::Mat right;
    makeShiftedPair(12, left, right);

    StereoMatcherParams params;
    params.type = "SGBM";
    params.num_disparities = 32;
    params.block_size = 5;
    params.num_tiles = 4;
    StereoMatcher matcher(params);

    const cv::Mat disparity = matcher.computeDisparity(left, right);
    const cv::Rect roi(60, 40, 200, 160);
    const cv::Mat disp_roi = disparity(roi);
    const cv::Mat mask = StereoMatcher::validMask(disp_roi);
    ASSERT_GT(cv::countNonZero(mask), roi.area() / 2);
    EXPECT_NEAR(cv::mean(disp_roi, mask)[0], 12.0, 1.5);
}

// The whole point of the overlap margin: tiling should be near-invisible to
// the output, not just "still roughly right". Compares num_tiles=1 against
// num_tiles=6 pixel-by-pixel on the same pair.
TEST(StereoMatcher, TiledMatchesUntiledClosely) {
    cv::Mat left;
    cv::Mat right;
    makeShiftedPair(12, left, right);

    StereoMatcherParams params;
    params.type = "SGBM";
    params.num_disparities = 32;
    params.block_size = 5;

    params.num_tiles = 1;
    const cv::Mat untiled = StereoMatcher(params).computeDisparity(left, right);
    params.num_tiles = 6;
    const cv::Mat tiled = StereoMatcher(params).computeDisparity(left, right);

    const cv::Mat mask = StereoMatcher::validMask(untiled) & StereoMatcher::validMask(tiled);
    const int valid = cv::countNonZero(mask);
    ASSERT_GT(valid, untiled.total() / 2);

    cv::Mat diff;
    cv::absdiff(untiled, tiled, diff);
    diff.setTo(0, ~mask);
    const double mean_abs_diff = cv::sum(diff)[0] / static_cast<double>(valid);
    EXPECT_LT(mean_abs_diff, 0.5) << "tiling should barely perturb the result at valid pixels";

    double max_diff = 0.0;
    cv::minMaxLoc(diff, nullptr, &max_diff, nullptr, nullptr, mask);
    EXPECT_LT(max_diff, 8.0) << "no single pixel should be wildly off just for sitting near a seam";
}

TEST(StereoMatcher, NumTilesDoesNotCrashWithManyTilesOnASmallImage) {
    cv::Mat left;
    cv::Mat right;
    makeShiftedPair(8, left, right);  // 240x320

    StereoMatcherParams params;
    params.type = "SGBM";
    params.num_disparities = 32;
    params.num_tiles = 40;  // deliberately many, several rows per tile
    StereoMatcher matcher(params);
    EXPECT_NO_THROW(matcher.computeDisparity(left, right));
}

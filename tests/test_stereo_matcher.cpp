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

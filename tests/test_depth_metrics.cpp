#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include "s3m/depth/depth_metrics.hpp"

using namespace s3m;

TEST(DepthMetrics, PerfectEstimate) {
    cv::Mat gt(20, 20, CV_32F, cv::Scalar::all(10.0));
    const cv::Mat est = gt.clone();

    const DepthMetrics m = evaluate(est, gt, 0.1, 100.0, {1.0, 2.0});
    EXPECT_NEAR(m.rmse, 0.0, 1e-9);
    EXPECT_NEAR(m.mae, 0.0, 1e-9);
    EXPECT_NEAR(m.density, 1.0, 1e-9);
    EXPECT_NEAR(m.delta1, 1.0, 1e-9);
    EXPECT_EQ(m.evaluated_pixels, 400);
}

TEST(DepthMetrics, KnownError) {
    cv::Mat gt(1, 4, CV_32F, cv::Scalar::all(10.0));
    cv::Mat est = gt.clone();
    est.at<float>(0, 0) = 13.0f;  // error 3
    est.at<float>(0, 1) = 11.0f;  // error 1

    const DepthMetrics m = evaluate(est, gt, 0.1, 100.0, {2.0});
    EXPECT_NEAR(m.mae, 1.0, 1e-6);                 // (3 + 1 + 0 + 0) / 4
    EXPECT_NEAR(m.rmse, std::sqrt(2.5), 1e-6);     // sqrt((9 + 1) / 4)
    ASSERT_EQ(m.bad_fraction.size(), 1u);
    EXPECT_NEAR(m.bad_fraction[0], 0.25, 1e-6);    // only the error-3 pixel exceeds 2.0
}

TEST(DepthMetrics, IgnoresInvalidPixels) {
    cv::Mat gt(1, 4, CV_32F);
    gt.at<float>(0, 0) = 5.0f;
    gt.at<float>(0, 1) = 0.0f;  // invalid ground truth
    gt.at<float>(0, 2) = 5.0f;
    gt.at<float>(0, 3) = 5.0f;

    cv::Mat est(1, 4, CV_32F, cv::Scalar::all(5.0));
    est.at<float>(0, 2) = std::numeric_limits<float>::quiet_NaN();  // invalid estimate

    const DepthMetrics m = evaluate(est, gt, 0.1, 100.0, {1.0});
    EXPECT_EQ(m.gt_valid_pixels, 3);
    EXPECT_EQ(m.evaluated_pixels, 2);
    EXPECT_NEAR(m.density, 2.0 / 3.0, 1e-6);
}

TEST(DepthMetrics, ToStringNonEmpty) {
    cv::Mat gt(4, 4, CV_32F, cv::Scalar::all(3.0));
    const DepthMetrics m = evaluate(gt, gt, 0.1, 100.0);
    EXPECT_FALSE(m.toString().empty());
}

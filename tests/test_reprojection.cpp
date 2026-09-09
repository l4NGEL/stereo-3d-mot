#include <gtest/gtest.h>

#include "s3m/depth/stereo_matcher.hpp"
#include "s3m/geometry/reprojection.hpp"

using namespace s3m;

TEST(Reprojection, ConstantDisparityPlane) {
    const StereoRig rig = StereoRig::fromIntrinsics(500, 500, 160, 120, cv::Size(320, 240), 0.1);
    cv::Mat disparity(240, 320, CV_32F, cv::Scalar::all(20.0));

    cv::Mat valid;
    const cv::Mat cloud = reproject(disparity, rig, &valid);

    EXPECT_EQ(cv::countNonZero(valid), 320 * 240);
    const cv::Vec3f centre = cloud.at<cv::Vec3f>(120, 160);
    EXPECT_NEAR(centre[2], 500.0 * 0.1 / 20.0, 1e-4);  // 2.5 m
    EXPECT_NEAR(centre[0], 0.0, 1e-4);
    EXPECT_NEAR(centre[1], 0.0, 1e-4);
}

TEST(Reprojection, InvalidDisparityProducesNoPoint) {
    const StereoRig rig = StereoRig::fromIntrinsics(500, 500, 160, 120, cv::Size(320, 240), 0.1);
    cv::Mat disparity(10, 10, CV_32F, cv::Scalar::all(StereoMatcher::kInvalidDisparity));
    cv::Mat valid;
    reproject(disparity, rig, &valid);
    EXPECT_EQ(cv::countNonZero(valid), 0);
}

TEST(Reprojection, DepthMapMatchesFormula) {
    const StereoRig rig = StereoRig::fromIntrinsics(400, 400, 100, 100, cv::Size(200, 200), 0.2);
    cv::Mat disparity(50, 50, CV_32F, cv::Scalar::all(16.0));
    const cv::Mat depth = disparityToDepthMap(disparity, rig);
    EXPECT_NEAR(depth.at<float>(25, 25), 400.0 * 0.2 / 16.0, 1e-3);  // 5 m
}

TEST(Reprojection, RobustDepthInRoiIgnoresOutliers) {
    cv::Mat depth(100, 100, CV_32F, cv::Scalar::all(5.0));
    depth(cv::Rect(0, 0, 100, 20)).setTo(50.0);  // 20 % gross outliers
    const float z = robustDepthInRoi(depth, cv::Rect(0, 0, 100, 100), 0.0f, 0.05f);
    EXPECT_NEAR(z, 5.0f, 1e-3);
}

TEST(Reprojection, RobustDepthInRoiRejectsSparseRoi) {
    cv::Mat depth(50, 50, CV_32F, cv::Scalar::all(0.0));  // all invalid
    EXPECT_LT(robustDepthInRoi(depth, cv::Rect(0, 0, 50, 50)), 0.0f);
}

TEST(Reprojection, PromoteTo3DUsesBoxCentre) {
    const StereoRig rig = StereoRig::fromIntrinsics(500, 500, 160, 120, cv::Size(320, 240), 0.1);
    cv::Mat depth(240, 320, CV_32F, cv::Scalar::all(4.0));

    Detection2D det;
    det.box = cv::Rect2f(140.0f, 100.0f, 40.0f, 40.0f);  // centre (160,120) = principal point
    det.score = 0.9f;

    const std::vector<Detection3D> out = promoteTo3D({det}, depth, rig);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_TRUE(out[0].valid);
    EXPECT_NEAR(out[0].depth, 4.0f, 1e-3);
    EXPECT_NEAR(out[0].position.x, 0.0f, 1e-3);
    EXPECT_NEAR(out[0].position.y, 0.0f, 1e-3);
    EXPECT_NEAR(out[0].position.z, 4.0f, 1e-3);
}

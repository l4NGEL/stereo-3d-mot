#include <cmath>
#include <stdexcept>

#include <gtest/gtest.h>

#include "s3m/camera/stereo_rig.hpp"

using s3m::StereoRig;

TEST(StereoRig, DisparityDepthRoundTrip) {
    const StereoRig rig = StereoRig::fromIntrinsics(700, 700, 320, 240, cv::Size(640, 480), 0.12);
    const double depths[] = {1.0, 2.5, 5.0, 10.0, 40.0};
    for (double z : depths) {
        const double d = rig.depthToDisparity(z);
        EXPECT_NEAR(rig.disparityToDepth(d), z, 1e-6);
    }
}

TEST(StereoRig, KnownDisparity) {
    // f = 700, B = 0.1, Z = 7  ->  d = 700 * 0.1 / 7 = 10 px
    const StereoRig rig = StereoRig::fromIntrinsics(700, 700, 320, 240, cv::Size(640, 480), 0.1);
    EXPECT_NEAR(rig.depthToDisparity(7.0), 10.0, 1e-9);
    EXPECT_NEAR(rig.disparityToDepth(10.0), 7.0, 1e-9);
}

TEST(StereoRig, TriangulateMatchesBackProjection) {
    const StereoRig rig = StereoRig::fromIntrinsics(650, 650, 320, 240, cv::Size(640, 480), 0.15);
    const cv::Point2d pixel(400.0, 300.0);
    const double z = 6.0;
    const cv::Point3d p = rig.triangulate(pixel, rig.depthToDisparity(z));
    EXPECT_NEAR(p.z, z, 1e-6);
    EXPECT_NEAR(p.x, (400.0 - 320.0) * z / 650.0, 1e-6);
    EXPECT_NEAR(p.y, (300.0 - 240.0) * z / 650.0, 1e-6);
}

TEST(StereoRig, ReprojectionMatrixConsistentWithTriangulate) {
    const StereoRig rig = StereoRig::fromIntrinsics(600, 600, 310, 250, cv::Size(640, 480), 0.2);
    const cv::Matx44d q = rig.reprojectionMatrix();
    const double u = 420.0;
    const double v = 190.0;
    const double d = 24.0;
    const cv::Vec4d h = q * cv::Vec4d(u, v, d, 1.0);
    const cv::Point3d from_q(h[0] / h[3], h[1] / h[3], h[2] / h[3]);
    const cv::Point3d from_tri = rig.triangulate(cv::Point2d(u, v), d);
    EXPECT_NEAR(from_q.x, from_tri.x, 1e-6);
    EXPECT_NEAR(from_q.y, from_tri.y, 1e-6);
    EXPECT_NEAR(from_q.z, from_tri.z, 1e-6);
}

TEST(StereoRig, DoffsShiftsDepth) {
    StereoRig rig = StereoRig::fromIntrinsics(600, 600, 320, 240, cv::Size(640, 480), 0.1);
    const double z0 = rig.disparityToDepth(20.0);
    rig.setDoffs(5.0);
    const double z1 = rig.disparityToDepth(20.0);
    EXPECT_LT(z1, z0);
    EXPECT_NEAR(z1, 600.0 * 0.1 / 25.0, 1e-9);
}

TEST(StereoRig, NonPositiveDisparityGivesInfiniteDepth) {
    const StereoRig rig = StereoRig::fromIntrinsics(600, 600, 320, 240, cv::Size(640, 480), 0.1);
    EXPECT_FALSE(std::isfinite(rig.disparityToDepth(0.0)));
    EXPECT_FALSE(std::isfinite(rig.disparityToDepth(-3.0)));
}

TEST(StereoRig, BaselineMustBePositive) {
    EXPECT_THROW(StereoRig::fromIntrinsics(600, 600, 320, 240, cv::Size(640, 480), 0.0),
                 std::invalid_argument);
}

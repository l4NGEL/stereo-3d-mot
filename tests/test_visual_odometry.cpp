#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include "s3m/vo/visual_odometry.hpp"

using namespace s3m;

namespace {

StereoRig testRig() {
    return StereoRig::fromIntrinsics(600.0, 600.0, 320.0, 240.0, cv::Size(640, 480), 0.12);
}

}  // namespace

// The full ORB-match -> essential-matrix -> stereo-scale pipeline needs real,
// trackable image content to exercise meaningfully -- covered by the real
// KITTI odometry run (docs/roadmap.md Phase 7), which is a more convincing
// end-to-end check than a synthetic two-image test would be, and by the
// Python validation of the underlying pose/scale math before it was ported
// (same doc). This test covers VisualOdometry's own state-machine contract,
// which doesn't depend on image content: the first frame is always the
// trajectory's origin, and reset() actually resets.
TEST(VisualOdometry, FirstFrameIsAlwaysTheIdentityOrigin) {
    VisualOdometry vo(testRig());
    const cv::Mat left(480, 640, CV_8UC3, cv::Scalar(80, 80, 80));
    const cv::Mat right(480, 640, CV_8UC3, cv::Scalar(80, 80, 80));

    const VoFrameResult r0 = vo.processFrame(left, right);
    EXPECT_TRUE(r0.ok);
    EXPECT_NEAR(r0.pose.position.x, 0.0, 1e-9);
    EXPECT_NEAR(r0.pose.position.y, 0.0, 1e-9);
    EXPECT_NEAR(r0.pose.position.z, 0.0, 1e-9);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) EXPECT_NEAR(r0.pose.rotation(r, c), (r == c) ? 1.0 : 0.0, 1e-9);
}

// A featureless (flat-color) frame pair can't produce any ORB matches, so
// the second frame must coast (hold the last pose, ok=false) instead of
// crashing or fabricating a pose from zero evidence.
TEST(VisualOdometry, FeaturelessFramesCoastRatherThanFail) {
    VisualOdometry vo(testRig());
    const cv::Mat flat(480, 640, CV_8UC3, cv::Scalar(80, 80, 80));

    const VoFrameResult r0 = vo.processFrame(flat, flat);
    ASSERT_TRUE(r0.ok);
    const VoFrameResult r1 = vo.processFrame(flat, flat);
    EXPECT_FALSE(r1.ok);
    EXPECT_NEAR(r1.pose.position.x, r0.pose.position.x, 1e-9);
    EXPECT_NEAR(r1.pose.position.y, r0.pose.position.y, 1e-9);
    EXPECT_NEAR(r1.pose.position.z, r0.pose.position.z, 1e-9);
}

TEST(VisualOdometry, ResetReturnsToIdentityOrigin) {
    VisualOdometry vo(testRig());
    const cv::Mat left(480, 640, CV_8UC3, cv::Scalar(80, 80, 80));
    const cv::Mat right(480, 640, CV_8UC3, cv::Scalar(80, 80, 80));

    vo.processFrame(left, right);
    vo.processFrame(left, right);
    vo.reset();

    const VoFrameResult r = vo.processFrame(left, right);
    EXPECT_TRUE(r.ok);
    EXPECT_NEAR(r.pose.position.x, 0.0, 1e-9);
    EXPECT_NEAR(r.pose.position.y, 0.0, 1e-9);
    EXPECT_NEAR(r.pose.position.z, 0.0, 1e-9);
}

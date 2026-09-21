#include <vector>

#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include "s3m/tracking/appearance.hpp"

using namespace s3m;

namespace {

cv::Mat solidColor(int size, cv::Scalar bgr) {
    return cv::Mat(size, size, CV_8UC3, bgr);
}

}  // namespace

TEST(Appearance, IdenticalCropsHaveZeroDistance) {
    const cv::Mat img = solidColor(40, cv::Scalar(0, 0, 255));  // pure red
    const cv::Mat a = computeAppearanceDescriptor(img, cv::Rect2f(0, 0, 40, 40));
    const cv::Mat b = computeAppearanceDescriptor(img, cv::Rect2f(0, 0, 40, 40));
    ASSERT_FALSE(a.empty());
    EXPECT_NEAR(appearanceDistance(a, b), 0.0, 1e-9);
}

// Pure red/green/blue land in different hue histogram bins with zero overlap
// -- Bhattacharyya distance between disjoint-support normalised histograms is
// exactly 1.0, not just "large", so this is an exact regression pin.
TEST(Appearance, DisjointHuesAreMaximallyDistant) {
    const cv::Mat red = solidColor(40, cv::Scalar(0, 0, 255));
    const cv::Mat green = solidColor(40, cv::Scalar(0, 255, 0));
    const cv::Mat blue = solidColor(40, cv::Scalar(255, 0, 0));

    const cv::Mat dr = computeAppearanceDescriptor(red, cv::Rect2f(0, 0, 40, 40));
    const cv::Mat dg = computeAppearanceDescriptor(green, cv::Rect2f(0, 0, 40, 40));
    const cv::Mat db = computeAppearanceDescriptor(blue, cv::Rect2f(0, 0, 40, 40));
    ASSERT_FALSE(dr.empty());
    ASSERT_FALSE(dg.empty());
    ASSERT_FALSE(db.empty());

    EXPECT_NEAR(appearanceDistance(dr, dg), 1.0, 1e-9);
    EXPECT_NEAR(appearanceDistance(dr, db), 1.0, 1e-9);
    EXPECT_NEAR(appearanceDistance(dg, db), 1.0, 1e-9);
}

TEST(Appearance, BoxOutsideImageReturnsEmptyDescriptor) {
    const cv::Mat img = solidColor(40, cv::Scalar(0, 0, 255));
    EXPECT_TRUE(computeAppearanceDescriptor(img, cv::Rect2f(1000, 1000, 40, 40)).empty());
}

TEST(Appearance, DegenerateBoxReturnsEmptyDescriptor) {
    const cv::Mat img = solidColor(40, cv::Scalar(0, 0, 255));
    EXPECT_TRUE(computeAppearanceDescriptor(img, cv::Rect2f(5, 5, 0, 0)).empty());
    EXPECT_TRUE(computeAppearanceDescriptor(img, cv::Rect2f(5, 5, -10, 10)).empty());
}

TEST(Appearance, EmptyImageReturnsEmptyDescriptor) {
    EXPECT_TRUE(computeAppearanceDescriptor(cv::Mat(), cv::Rect2f(0, 0, 10, 10)).empty());
}

TEST(Appearance, PartiallyOutOfBoundsBoxIsClippedNotRejected) {
    const cv::Mat img = solidColor(40, cv::Scalar(0, 0, 255));
    // Half the box is outside the 40x40 image -- should still yield a
    // descriptor from the overlapping half, not an empty one.
    EXPECT_FALSE(computeAppearanceDescriptor(img, cv::Rect2f(20, 20, 40, 40)).empty());
}

TEST(Appearance, AttachAppearanceFillsValidBoxesLeavesOutOfBoundsEmpty) {
    const cv::Mat img = solidColor(100, cv::Scalar(0, 0, 255));
    std::vector<Detection3D> dets(2);
    dets[0].box = cv::Rect2f(10, 10, 20, 20);    // inside
    dets[1].box = cv::Rect2f(500, 500, 20, 20);  // outside

    attachAppearance(dets, img);
    EXPECT_FALSE(dets[0].appearance.empty());
    EXPECT_TRUE(dets[1].appearance.empty());
}

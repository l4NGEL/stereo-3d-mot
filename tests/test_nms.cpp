#include <vector>

#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include "s3m/detection/nms.hpp"

using namespace s3m;

TEST(Iou, IdenticalDisjointAndHalfOverlap) {
    EXPECT_FLOAT_EQ(iou({0, 0, 10, 10}, {0, 0, 10, 10}), 1.0f);
    EXPECT_FLOAT_EQ(iou({0, 0, 10, 10}, {100, 100, 10, 10}), 0.0f);
    EXPECT_NEAR(iou({0, 0, 10, 10}, {5, 0, 10, 10}), 50.0f / 150.0f, 1e-5);
}

TEST(Nms, SuppressesOverlapsKeepsDisjoint) {
    const std::vector<cv::Rect2f> boxes{
        {0, 0, 10, 10}, {1, 1, 10, 10}, {100, 100, 10, 10}, {0, 0, 10, 10}};
    const std::vector<float> scores{0.9f, 0.8f, 0.7f, 0.95f};

    const std::vector<int> keep = nms(boxes, scores, 0.5f);
    ASSERT_EQ(keep.size(), 2u);
    EXPECT_EQ(keep[0], 3);  // top score first
    EXPECT_EQ(keep[1], 2);  // disjoint box survives
}

TEST(Nms, HighIouThresholdKeepsNearDuplicates) {
    const std::vector<cv::Rect2f> boxes{{0, 0, 10, 10}, {1, 1, 10, 10}, {100, 100, 10, 10}};
    const std::vector<float> scores{0.9f, 0.8f, 0.7f};
    EXPECT_EQ(nms(boxes, scores, 0.95f).size(), 3u);
}

TEST(Nms, ScoreThresholdDropsWeakCandidates) {
    const std::vector<cv::Rect2f> boxes{{0, 0, 10, 10}, {50, 50, 10, 10}};
    const std::vector<float> scores{0.9f, 0.1f};

    const std::vector<int> keep = nms(boxes, scores, 0.5f, /*score_threshold=*/0.3f);
    ASSERT_EQ(keep.size(), 1u);
    EXPECT_EQ(keep[0], 0);
}

TEST(Nms, ClassAwareSeparatesClasses) {
    const std::vector<cv::Rect2f> boxes{{0, 0, 10, 10}, {0, 0, 10, 10}};
    const std::vector<float> scores{0.9f, 0.8f};

    EXPECT_EQ(nmsClassAware(boxes, scores, {0, 1}, 0.5f).size(), 2u);  // different class: both kept
    EXPECT_EQ(nmsClassAware(boxes, scores, {7, 7}, 0.5f).size(), 1u);  // same class: suppressed
}

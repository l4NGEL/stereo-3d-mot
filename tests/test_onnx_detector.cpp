#if defined(S3M_WITH_ONNX)

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include "s3m/detection/onnx_detector.hpp"

using namespace s3m;

namespace {
std::string testModel() {
    return std::string(S3M_TEST_DATA_DIR) + "/tiny_yolov8.onnx";
}
}  // namespace

// The fixture model (scripts/make_test_model.py) ignores pixels and emits three
// raw boxes in 640x640 letterboxed space:
//   col 20: cx=200 cy=160 w=80  h=120  class 0  score 0.92
//   col 40: cx=450 cy=300 w=100 h=90   class 2  score 0.75
//   col 41: cx=452 cy=302 w=98  h=92   class 2  score 0.70  (NMS duplicate)

TEST(OnnxDetector, ParsesBoxesInvertsLetterboxAndRunsNms) {
    OnnxDetector::Options o;
    o.model_path = testModel();
    o.input_size = 640;
    o.score_threshold = 0.25f;
    o.nms_iou = 0.45f;
    OnnxDetector det(std::move(o));

    // 640x480 input -> letterbox scale 1.0, pad_x 0, pad_y 80.
    const cv::Mat img(480, 640, CV_8UC3, cv::Scalar::all(0));
    const std::vector<Detection2D> dets = det.detect(img);

    ASSERT_EQ(dets.size(), 2u);  // col 41 is suppressed against col 40

    EXPECT_EQ(dets[0].class_id, 0);
    EXPECT_NEAR(dets[0].score, 0.92f, 1e-4);
    EXPECT_NEAR(dets[0].box.x, 160.0f, 1.0f);  // (200 - 40) - 0
    EXPECT_NEAR(dets[0].box.y, 20.0f, 1.0f);   // (160 - 60) - 80
    EXPECT_NEAR(dets[0].box.width, 80.0f, 1.0f);
    EXPECT_NEAR(dets[0].box.height, 120.0f, 1.0f);

    EXPECT_EQ(dets[1].class_id, 2);
    EXPECT_NEAR(dets[1].score, 0.75f, 1e-4);
    EXPECT_NEAR(dets[1].box.x, 400.0f, 1.0f);  // (450 - 50) - 0
    EXPECT_NEAR(dets[1].box.y, 175.0f, 1.0f);  // (300 - 45) - 80
}

TEST(OnnxDetector, KeepClassesFiltersOutput) {
    OnnxDetector::Options o;
    o.model_path = testModel();
    o.keep_classes = {2};
    OnnxDetector det(std::move(o));

    const cv::Mat img(480, 640, CV_8UC3, cv::Scalar::all(0));
    const std::vector<Detection2D> dets = det.detect(img);

    ASSERT_EQ(dets.size(), 1u);
    EXPECT_EQ(dets[0].class_id, 2);
}

TEST(OnnxDetector, HighScoreThresholdRejectsWeakBoxes) {
    OnnxDetector::Options o;
    o.model_path = testModel();
    o.score_threshold = 0.85f;  // only class 0 (0.92) clears this
    OnnxDetector det(std::move(o));

    const cv::Mat img(480, 640, CV_8UC3, cv::Scalar::all(0));
    const std::vector<Detection2D> dets = det.detect(img);

    ASSERT_EQ(dets.size(), 1u);
    EXPECT_EQ(dets[0].class_id, 0);
}

TEST(OnnxDetector, MissingModelThrows) {
    OnnxDetector::Options o;
    o.model_path = "/no/such/model.onnx";
    EXPECT_ANY_THROW(OnnxDetector det(std::move(o)));
}

TEST(OnnxDetector, EmptyModelPathThrows) {
    OnnxDetector::Options o;
    EXPECT_THROW(OnnxDetector det(std::move(o)), std::invalid_argument);
}

#endif  // S3M_WITH_ONNX

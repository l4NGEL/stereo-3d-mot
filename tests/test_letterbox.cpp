#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include "s3m/detection/letterbox.hpp"

using namespace s3m;

TEST(Letterbox, WideImagePadsVertically) {
    cv::Mat src(480, 640, CV_8UC3, cv::Scalar(10, 20, 30));
    cv::Mat dst;
    const Letterbox lb = makeLetterbox(src, dst, 640);

    EXPECT_EQ(dst.cols, 640);
    EXPECT_EQ(dst.rows, 640);
    EXPECT_FLOAT_EQ(lb.scale, 1.0f);
    EXPECT_EQ(lb.pad_x, 0);
    EXPECT_EQ(lb.pad_y, 80);  // (640 - 480) / 2
    EXPECT_EQ(lb.scaled.width, 640);
    EXPECT_EQ(lb.scaled.height, 480);
}

TEST(Letterbox, PortraitImagePadsHorizontally) {
    cv::Mat src(640, 480, CV_8UC3, cv::Scalar::all(0));
    cv::Mat dst;
    const Letterbox lb = makeLetterbox(src, dst, 320);

    EXPECT_FLOAT_EQ(lb.scale, 0.5f);
    EXPECT_EQ(lb.scaled.width, 240);
    EXPECT_EQ(lb.scaled.height, 320);
    EXPECT_EQ(lb.pad_x, 40);
    EXPECT_EQ(lb.pad_y, 0);
}

TEST(Letterbox, InverseMapsBoxBackToOriginal) {
    cv::Mat src(720, 1280, CV_8UC3, cv::Scalar::all(0));
    cv::Mat dst;
    const Letterbox lb = makeLetterbox(src, dst, 640);

    const cv::Rect2f orig(100.0f, 50.0f, 60.0f, 90.0f);
    const cv::Rect2f letterboxed(orig.x * lb.scale + static_cast<float>(lb.pad_x),
                                 orig.y * lb.scale + static_cast<float>(lb.pad_y),
                                 orig.width * lb.scale, orig.height * lb.scale);
    const cv::Rect2f back = lb.toOriginal(letterboxed);

    EXPECT_NEAR(back.x, orig.x, 1e-3);
    EXPECT_NEAR(back.y, orig.y, 1e-3);
    EXPECT_NEAR(back.width, orig.width, 1e-3);
    EXPECT_NEAR(back.height, orig.height, 1e-3);
}

TEST(Letterbox, ContentGoesInsidePaddingBorder) {
    cv::Mat src(100, 200, CV_8UC1, cv::Scalar::all(255));
    cv::Mat dst;
    const Letterbox lb = makeLetterbox(src, dst, 64, cv::Scalar::all(0));

    EXPECT_EQ(lb.pad_x, 0);
    EXPECT_EQ(lb.pad_y, 16);            // scaled 64x32, centred in 64
    EXPECT_EQ(dst.at<uchar>(0, 0), 0);    // top padding
    EXPECT_EQ(dst.at<uchar>(32, 32), 255);  // content
}

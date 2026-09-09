#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "s3m/io/pfm.hpp"

using namespace s3m;

TEST(Pfm, WriteReadRoundTrip) {
    cv::Mat image(7, 11, CV_32F);
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            image.at<float>(y, x) = static_cast<float>(y * 100 + x) + 0.5f;
        }
    }

    const std::string path = "s3m_test_roundtrip.pfm";
    writePfm(path, image);
    const cv::Mat back = readPfm(path);
    std::remove(path.c_str());

    ASSERT_EQ(back.type(), CV_32FC1);
    ASSERT_EQ(back.rows, image.rows);
    ASSERT_EQ(back.cols, image.cols);
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            EXPECT_FLOAT_EQ(back.at<float>(y, x), image.at<float>(y, x));
        }
    }
}

TEST(Pfm, PreservesRowOrder) {
    cv::Mat image = cv::Mat::zeros(4, 4, CV_32F);
    image.at<float>(0, 0) = 1.0f;  // top-left marker
    image.at<float>(3, 3) = 9.0f;  // bottom-right marker

    const std::string path = "s3m_test_roworder.pfm";
    writePfm(path, image);
    const cv::Mat back = readPfm(path);
    std::remove(path.c_str());

    EXPECT_FLOAT_EQ(back.at<float>(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(back.at<float>(3, 3), 9.0f);
}

TEST(Pfm, RejectsNonPfmFile) {
    const std::string path = "s3m_test_not_a.pfm";
    {
        std::ofstream f(path);
        f << "this is not a pfm file";
    }
    EXPECT_THROW(readPfm(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(Pfm, RejectsMissingFile) {
    EXPECT_THROW(readPfm("s3m_definitely_missing_98765.pfm"), std::runtime_error);
}

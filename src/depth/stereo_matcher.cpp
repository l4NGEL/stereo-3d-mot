#include "s3m/depth/stereo_matcher.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <string>

#include <opencv2/imgproc.hpp>

namespace s3m {
namespace {

int makeOdd(int value, int lo, int hi) {
    value = std::clamp(value, lo, hi);
    if (value % 2 == 0) ++value;
    return value;
}

int roundUpTo16(int value) { return value <= 16 ? 16 : ((value + 15) / 16) * 16; }

std::string upper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return s;
}

cv::Mat toGray(const cv::Mat& image) {
    if (image.channels() == 1) return image;
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

}  // namespace

StereoMatcher::StereoMatcher(const StereoMatcherParams& params) { configure(params); }

void StereoMatcher::configure(const StereoMatcherParams& params) {
    params_ = params;
    params_.num_disparities = roundUpTo16(params_.num_disparities);

    const std::string type = upper(params_.type);
    const int pre_cap = std::clamp(params_.pre_filter_cap, 1, 63);
    const int uniqueness = std::max(0, params_.uniqueness_ratio);
    const int speckle_win = std::max(0, params_.speckle_window_size);
    const int speckle_range = std::max(0, params_.speckle_range);

    if (type == "BM") {
        const int block = makeOdd(params_.block_size, 5, 51);
        cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(params_.num_disparities, block);
        bm->setMinDisparity(params_.min_disparity);
        bm->setPreFilterCap(pre_cap);
        bm->setUniquenessRatio(uniqueness);
        bm->setSpeckleWindowSize(speckle_win);
        bm->setSpeckleRange(speckle_range);
        bm->setDisp12MaxDiff(params_.disp12_max_diff);
        matcher_ = bm;
    } else if (type == "SGBM") {
        const int block = makeOdd(params_.block_size, 1, 11);
        const int p1 = params_.p1_multiplier * block * block;
        const int p2 = std::max(p1 + 1, params_.p2_multiplier * block * block);
        matcher_ = cv::StereoSGBM::create(
            params_.min_disparity, params_.num_disparities, block, p1, p2,
            params_.disp12_max_diff, pre_cap, uniqueness, speckle_win, speckle_range,
            params_.mode_hh ? cv::StereoSGBM::MODE_HH : cv::StereoSGBM::MODE_SGBM);
    } else {
        throw std::invalid_argument("StereoMatcher: unknown type '" + params_.type + "'");
    }
}

cv::Mat StereoMatcher::computeDisparity(const cv::Mat& left, const cv::Mat& right) {
    if (left.empty() || right.empty() || left.size() != right.size()) {
        throw std::invalid_argument("StereoMatcher::computeDisparity: empty or mismatched inputs");
    }
    if (!matcher_) {
        throw std::runtime_error("StereoMatcher::computeDisparity: matcher not configured");
    }

    cv::Mat raw;
    matcher_->compute(toGray(left), toGray(right), raw);

    // StereoBM / StereoSGBM emit CV_16S fixed point (true disparity = raw / 16).
    cv::Mat disparity;
    if (raw.type() == CV_16S) {
        raw.convertTo(disparity, CV_32F, 1.0 / 16.0);
    } else {
        raw.convertTo(disparity, CV_32F);
    }

    const float min_valid = static_cast<float>(params_.min_disparity);
    for (int y = 0; y < disparity.rows; ++y) {
        float* row = disparity.ptr<float>(y);
        for (int x = 0; x < disparity.cols; ++x) {
            if (!std::isfinite(row[x]) || row[x] < min_valid) row[x] = kInvalidDisparity;
        }
    }
    return disparity;
}

cv::Mat StereoMatcher::validMask(const cv::Mat& disparity) {
    CV_Assert(disparity.type() == CV_32F);
    cv::Mat mask(disparity.size(), CV_8U);
    for (int y = 0; y < disparity.rows; ++y) {
        const float* src = disparity.ptr<float>(y);
        uchar* dst = mask.ptr<uchar>(y);
        for (int x = 0; x < disparity.cols; ++x) {
            const float value = src[x];
            dst[x] = (std::isfinite(value) && value != kInvalidDisparity) ? 255 : 0;
        }
    }
    return mask;
}

}  // namespace s3m

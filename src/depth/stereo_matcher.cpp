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
    if (value % 2 == 0)
        ++value;
    return value;
}

int roundUpTo16(int value) {
    return value <= 16 ? 16 : ((value + 15) / 16) * 16;
}

std::string upper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return s;
}

cv::Mat toGray(const cv::Mat& image) {
    if (image.channels() == 1)
        return image;
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

/// Builds one BM/SGBM instance from already-normalised params. Factored out
/// of configure() so the tiled path can build several independent instances
/// (one per tile) with exactly the same settings as the untiled one.
cv::Ptr<cv::StereoMatcher> makeMatcher(const StereoMatcherParams& p, const std::string& type,
                                       int pre_cap, int uniqueness, int speckle_win,
                                       int speckle_range) {
    if (type == "BM") {
        const int block = makeOdd(p.block_size, 5, 51);
        cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(p.num_disparities, block);
        bm->setMinDisparity(p.min_disparity);
        bm->setPreFilterCap(pre_cap);
        bm->setUniquenessRatio(uniqueness);
        bm->setSpeckleWindowSize(speckle_win);
        bm->setSpeckleRange(speckle_range);
        bm->setDisp12MaxDiff(p.disp12_max_diff);
        return bm;
    }
    if (type == "SGBM") {
        const int block = makeOdd(p.block_size, 1, 11);
        const int p1 = p.p1_multiplier * block * block;
        const int p2 = std::max(p1 + 1, p.p2_multiplier * block * block);
        return cv::StereoSGBM::create(
            p.min_disparity, p.num_disparities, block, p1, p2, p.disp12_max_diff, pre_cap,
            uniqueness, speckle_win, speckle_range,
            p.mode_hh ? cv::StereoSGBM::MODE_HH : cv::StereoSGBM::MODE_SGBM);
    }
    throw std::invalid_argument("StereoMatcher: unknown type '" + p.type + "'");
}

}  // namespace

StereoMatcher::StereoMatcher(const StereoMatcherParams& params) {
    configure(params);
}

void StereoMatcher::configure(const StereoMatcherParams& params) {
    params_ = params;
    params_.num_disparities = roundUpTo16(params_.num_disparities);

    const std::string type = upper(params_.type);
    const int pre_cap = std::clamp(params_.pre_filter_cap, 1, 63);
    const int uniqueness = std::max(0, params_.uniqueness_ratio);
    const int speckle_win = std::max(0, params_.speckle_window_size);
    const int speckle_range = std::max(0, params_.speckle_range);
    const int block_for_overlap =
        (type == "BM") ? makeOdd(params_.block_size, 5, 51) : makeOdd(params_.block_size, 1, 11);

    matcher_ = nullptr;
    tile_matchers_.clear();
    if (params_.num_tiles <= 1) {
        matcher_ = makeMatcher(params_, type, pre_cap, uniqueness, speckle_win, speckle_range);
    } else {
        tile_matchers_.reserve(static_cast<std::size_t>(params_.num_tiles));
        for (int i = 0; i < params_.num_tiles; ++i) {
            tile_matchers_.push_back(
                makeMatcher(params_, type, pre_cap, uniqueness, speckle_win, speckle_range));
        }
        // Margin so each tile's block-matching window and SGBM's aggregation
        // near the seam see close to the same neighbourhood the untiled
        // matcher would have. Deliberately tight, not the conservative
        // max(32, 8*block) first guess: on KITTI-sized (short, ~375-row)
        // images the overlap competes directly with tile height for a fixed
        // tile count, and an overly generous margin makes more tiles a net
        // *loss* (each tile redoes a larger fraction of its neighbours'
        // work). See docs/roadmap.md Phase 6 for the measured accuracy this
        // buys and the tile-count sweep that motivated tightening it.
        overlap_rows_ = std::max(16, 3 * block_for_overlap);
    }
}

cv::Mat StereoMatcher::computeTiledRaw(const cv::Mat& left_gray, const cv::Mat& right_gray) const {
    const int rows = left_gray.rows;
    const int n = static_cast<int>(tile_matchers_.size());
    cv::Mat raw(left_gray.size(), CV_16S);
    const int base_h = rows / n;

    cv::parallel_for_(cv::Range(0, n), [&](const cv::Range& r) {
        for (int i = r.start; i < r.end; ++i) {
            const int y0 = i * base_h;
            const int y1 = (i == n - 1) ? rows : (i + 1) * base_h;
            const int ov_top = std::min(overlap_rows_, y0);
            const int ov_bottom = std::min(overlap_rows_, rows - y1);
            const cv::Range padded(y0 - ov_top, y1 + ov_bottom);

            cv::Mat tile_raw;
            tile_matchers_[static_cast<std::size_t>(i)]->compute(
                left_gray.rowRange(padded), right_gray.rowRange(padded), tile_raw);
            tile_raw.rowRange(ov_top, ov_top + (y1 - y0)).copyTo(raw.rowRange(y0, y1));
        }
    });
    return raw;
}

cv::Mat StereoMatcher::computeDisparity(const cv::Mat& left, const cv::Mat& right) {
    if (left.empty() || right.empty() || left.size() != right.size()) {
        throw std::invalid_argument("StereoMatcher::computeDisparity: empty or mismatched inputs");
    }
    if (!matcher_ && tile_matchers_.empty()) {
        throw std::runtime_error("StereoMatcher::computeDisparity: matcher not configured");
    }

    const cv::Mat left_gray = toGray(left);
    const cv::Mat right_gray = toGray(right);
    cv::Mat raw;
    if (matcher_) {
        matcher_->compute(left_gray, right_gray, raw);
    } else {
        raw = computeTiledRaw(left_gray, right_gray);
    }

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
            if (!std::isfinite(row[x]) || row[x] < min_valid)
                row[x] = kInvalidDisparity;
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

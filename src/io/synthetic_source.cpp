#include "s3m/io/synthetic_source.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

#include <opencv2/imgproc.hpp>

namespace s3m {
namespace {

constexpr int kTextureMargin = 96;  ///< horizontal slack in the background texture (px)

/// Multi-scale texture: coloured noise around `tint` blended with an upsampled
/// low-frequency block pattern so block matching has features at every scale.
cv::Mat makeTexture(cv::Size size, std::uint64_t seed, const cv::Scalar& tint) {
    cv::RNG rng(seed);

    cv::Mat high(size, CV_8UC3);
    rng.fill(high, cv::RNG::NORMAL, tint, cv::Scalar::all(18));

    const cv::Size low_size(std::max(2, size.width / 24), std::max(2, size.height / 24));
    cv::Mat low(low_size, CV_8UC3);
    rng.fill(low, cv::RNG::UNIFORM, cv::Scalar::all(40), cv::Scalar::all(215));
    cv::Mat low_up;
    cv::resize(low, low_up, size, 0, 0, cv::INTER_LINEAR);

    cv::Mat out;
    cv::addWeighted(high, 0.55, low_up, 0.45, 0.0, out);
    return out;
}

void addSensorNoise(cv::Mat& image_bgr, std::uint64_t seed, double std_dev) {
    if (std_dev <= 0.0)
        return;
    cv::RNG rng(seed);
    cv::Mat noise(image_bgr.size(), CV_32FC3);
    rng.fill(noise, cv::RNG::NORMAL, 0.0, std_dev);
    cv::Mat as_float;
    image_bgr.convertTo(as_float, CV_32FC3);
    as_float += noise;
    as_float.convertTo(image_bgr, CV_8UC3);  // convertTo saturates
}

std::uint64_t mix(std::uint64_t a, std::uint64_t b) {
    return a * 0x9E3779B97F4A7C15ull + b + 0x1234567;
}

}  // namespace

std::vector<SyntheticStereoSource::Card> SyntheticStereoSource::Options::defaultCards() {
    std::vector<Card> cards;
    cards.push_back(Card{cv::Rect(110, 180, 95, 150), 2.5});  // near, drifts right
    cards.push_back(Card{cv::Rect(430, 150, 80, 130), 6.0});  // far-ish, drifts left
    cards.push_back(Card{cv::Rect(275, 250, 70, 110), 3.8});  // mid, drifts right
    return cards;
}

SyntheticStereoSource::SyntheticStereoSource(Options options) : options_(std::move(options)) {
    if (options_.cards.empty())
        options_.cards = Options::defaultCards();
    if (options_.cx <= 0.0)
        options_.cx = options_.image_size.width / 2.0;
    if (options_.cy <= 0.0)
        options_.cy = options_.image_size.height / 2.0;
    if (options_.num_frames < 1)
        options_.num_frames = 1;

    rig_ = StereoRig::fromIntrinsics(options_.fx, options_.fy, options_.cx, options_.cy,
                                     options_.image_size, options_.baseline_m);

    background_texture_ = makeTexture(
        cv::Size(options_.image_size.width + 2 * kTextureMargin, options_.image_size.height),
        mix(options_.seed, 1), cv::Scalar::all(128));

    static const cv::Scalar tints[] = {cv::Scalar(170, 120, 90), cv::Scalar(90, 175, 120),
                                       cv::Scalar(95, 105, 205), cv::Scalar(165, 165, 90),
                                       cv::Scalar(150, 90, 170)};
    card_textures_.reserve(options_.cards.size());
    for (std::size_t i = 0; i < options_.cards.size(); ++i) {
        const cv::Size sz(std::max(24, options_.cards[i].region.width),
                          std::max(24, options_.cards[i].region.height));
        card_textures_.push_back(makeTexture(sz, mix(options_.seed, 100 + i), tints[i % 5]));
    }
}

cv::Rect SyntheticStereoSource::cardRegionAt(int card_index, int frame_index) const {
    const Card& card = options_.cards[static_cast<std::size_t>(card_index)];
    const double direction = (card_index % 2 == 0) ? 1.0 : -1.0;
    const int dx = static_cast<int>(
        std::lround(options_.card_drift_px * static_cast<double>(frame_index) * direction));
    cv::Rect region = card.region + cv::Point(dx, 0);
    region &= cv::Rect(0, 0, options_.image_size.width, options_.image_size.height);
    return region;
}

std::vector<cv::Rect> SyntheticStereoSource::cardBoxes(int frame_index) const {
    std::vector<cv::Rect> boxes;
    boxes.reserve(options_.cards.size());
    for (std::size_t i = 0; i < options_.cards.size(); ++i) {
        boxes.push_back(cardRegionAt(static_cast<int>(i), frame_index));
    }
    return boxes;
}

StereoFrame SyntheticStereoSource::render(int frame_index) const {
    const int w = options_.image_size.width;
    const int h = options_.image_size.height;
    const double f_baseline = options_.fx * options_.baseline_m;

    // ---- background: same texture sampled into both views ----------------
    const double d_bg = f_baseline / options_.background_depth_m;
    const int bg_shift = std::clamp(static_cast<int>(std::lround(d_bg)), 0, kTextureMargin);

    cv::Mat left = background_texture_(cv::Rect(kTextureMargin, 0, w, h)).clone();
    cv::Mat right = background_texture_(cv::Rect(kTextureMargin + bg_shift, 0, w, h)).clone();

    cv::Mat gt_disparity(h, w, CV_32F, cv::Scalar::all(static_cast<float>(d_bg)));
    cv::Mat gt_depth(h, w, CV_32F,
                     cv::Scalar::all(static_cast<float>(options_.background_depth_m)));

    // ---- cards, painted far-to-near so nearer cards occlude -------------
    std::vector<int> order(options_.cards.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [this](int a, int b) {
        return options_.cards[static_cast<std::size_t>(a)].depth_m >
               options_.cards[static_cast<std::size_t>(b)].depth_m;
    });

    for (int idx : order) {
        const Card& card = options_.cards[static_cast<std::size_t>(idx)];
        const cv::Rect region = cardRegionAt(idx, frame_index);
        if (region.width <= 0 || region.height <= 0)
            continue;

        const double d_card = f_baseline / card.depth_m;
        const int shift = static_cast<int>(std::lround(d_card));

        cv::Mat card_tile;
        cv::resize(card_textures_[static_cast<std::size_t>(idx)], card_tile, region.size());

        card_tile.copyTo(left(region));

        const cv::Rect right_region(region.x - shift, region.y, region.width, region.height);
        const cv::Rect right_visible = right_region & cv::Rect(0, 0, w, h);
        if (right_visible.width > 0 && right_visible.height > 0) {
            const cv::Rect crop(right_visible.x - right_region.x, right_visible.y - right_region.y,
                                right_visible.width, right_visible.height);
            card_tile(crop).copyTo(right(right_visible));
        }

        gt_disparity(region).setTo(cv::Scalar::all(d_card));
        gt_depth(region).setTo(cv::Scalar::all(card.depth_m));
    }

    addSensorNoise(left, mix(options_.seed, mix(2, static_cast<std::uint64_t>(frame_index))),
                   options_.noise_std);
    addSensorNoise(right, mix(options_.seed, mix(3, static_cast<std::uint64_t>(frame_index))),
                   options_.noise_std);

    StereoFrame frame;
    frame.index = frame_index;
    frame.timestamp = static_cast<double>(frame_index) / 30.0;
    frame.left = std::move(left);
    frame.right = std::move(right);
    frame.gt_disparity = std::move(gt_disparity);
    frame.gt_depth = std::move(gt_depth);
    return frame;
}

std::optional<StereoFrame> SyntheticStereoSource::next() {
    if (cursor_ >= options_.num_frames)
        return std::nullopt;
    return render(cursor_++);
}

}  // namespace s3m

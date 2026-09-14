#include "s3m/tracking/appearance.hpp"

#include <opencv2/imgproc.hpp>

namespace s3m {
namespace {

constexpr int kHueBins = 16;
constexpr int kSatBins = 16;

}  // namespace

cv::Mat computeAppearanceDescriptor(const cv::Mat& image_bgr, const cv::Rect2f& box) {
    if (image_bgr.empty()) return cv::Mat();
    const cv::Rect roi = cv::Rect(box) & cv::Rect(cv::Point(0, 0), image_bgr.size());
    if (roi.width <= 0 || roi.height <= 0) return cv::Mat();

    cv::Mat hsv;
    cv::cvtColor(image_bgr(roi), hsv, cv::COLOR_BGR2HSV);

    const int channels[] = {0, 1};
    const int hist_size[] = {kHueBins, kSatBins};
    const float hue_range[] = {0.0f, 180.0f};
    const float sat_range[] = {0.0f, 256.0f};
    const float* ranges[] = {hue_range, sat_range};

    cv::Mat hist;
    cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 2, hist_size, ranges);
    cv::normalize(hist, hist, 1.0, 0.0, cv::NORM_L1);
    return hist;
}

double appearanceDistance(const cv::Mat& a, const cv::Mat& b) {
    return cv::compareHist(a, b, cv::HISTCMP_BHATTACHARYYA);
}

void attachAppearance(std::vector<Detection3D>& detections, const cv::Mat& image_bgr) {
    for (Detection3D& d : detections) d.appearance = computeAppearanceDescriptor(image_bgr, d.box);
}

}  // namespace s3m

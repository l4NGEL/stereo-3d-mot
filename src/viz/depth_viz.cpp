#include "s3m/viz/depth_viz.hpp"

#include <algorithm>
#include <cmath>
#include <string>

#include <opencv2/imgproc.hpp>

namespace s3m {
namespace {

cv::Mat normalizeToU8(const cv::Mat& src, float lo, float hi, cv::Mat& mask_out) {
    cv::Mat u8(src.size(), CV_8U, cv::Scalar::all(0));
    mask_out = cv::Mat(src.size(), CV_8U, cv::Scalar::all(0));
    const float span = (hi > lo) ? (hi - lo) : 1.0f;

    for (int y = 0; y < src.rows; ++y) {
        const float* s = src.ptr<float>(y);
        uchar* d = u8.ptr<uchar>(y);
        uchar* m = mask_out.ptr<uchar>(y);
        for (int x = 0; x < src.cols; ++x) {
            const float v = s[x];
            if (!std::isfinite(v) || v <= 0.0f) continue;
            const float t = std::clamp((v - lo) / span, 0.0f, 1.0f);
            d[x] = static_cast<uchar>(std::lround(t * 255.0f));
            m[x] = 255;
        }
    }
    return u8;
}

void blackOutInvalid(cv::Mat& color_bgr, const cv::Mat& valid_mask) {
    cv::Mat invalid;
    cv::bitwise_not(valid_mask, invalid);
    color_bgr.setTo(cv::Scalar::all(0), invalid);
}

cv::Mat ensureBgr(const cv::Mat& image) {
    if (image.channels() == 3) return image.clone();
    cv::Mat bgr;
    cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
    return bgr;
}

void putLabel(cv::Mat& img, const std::string& text, cv::Point origin, const cv::Scalar& color) {
    cv::putText(img, text, origin, cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
}

}  // namespace

cv::Mat colorizeDisparity(const cv::Mat& disparity, float max_disparity, int colormap) {
    CV_Assert(disparity.type() == CV_32F);
    cv::Mat mask;
    const cv::Mat u8 =
        normalizeToU8(disparity, 0.0f, max_disparity > 0.0f ? max_disparity : 64.0f, mask);
    cv::Mat color;
    cv::applyColorMap(u8, color, colormap);
    blackOutInvalid(color, mask);
    return color;
}

cv::Mat colorizeDepth(const cv::Mat& depth, float min_m, float max_m, int colormap) {
    CV_Assert(depth.type() == CV_32F);
    cv::Mat mask;
    cv::Mat u8 = normalizeToU8(depth, min_m, max_m, mask);
    cv::subtract(cv::Scalar::all(255), u8, u8);  // near -> warm end of the map
    cv::Mat color;
    cv::applyColorMap(u8, color, colormap);
    blackOutInvalid(color, mask);
    return color;
}

cv::Mat tile(const std::vector<cv::Mat>& images, int cols, int pad, const cv::Scalar& pad_color) {
    if (images.empty()) return {};
    cols = std::max(1, cols);
    const int rows = static_cast<int>((images.size() + static_cast<std::size_t>(cols) - 1) /
                                      static_cast<std::size_t>(cols));
    const cv::Size cell = images.front().size();

    cv::Mat canvas(rows * cell.height + (rows + 1) * pad, cols * cell.width + (cols + 1) * pad,
                   CV_8UC3, pad_color);

    for (std::size_t i = 0; i < images.size(); ++i) {
        if (images[i].empty()) continue;
        cv::Mat cellImg = ensureBgr(images[i]);
        if (cellImg.size() != cell) {
            cv::Mat resized;
            cv::resize(cellImg, resized, cell);
            cellImg = resized;
        }

        const int r = static_cast<int>(i) / cols;
        const int c = static_cast<int>(i) % cols;
        const int x = pad + c * (cell.width + pad);
        const int y = pad + r * (cell.height + pad);
        cellImg.copyTo(canvas(cv::Rect(x, y, cell.width, cell.height)));
    }
    return canvas;
}

cv::Mat drawDetections(const cv::Mat& image_bgr, const std::vector<Detection2D>& detections) {
    cv::Mat out = ensureBgr(image_bgr);
    for (const Detection2D& d : detections) {
        const cv::Point tl(cvRound(d.box.x), cvRound(d.box.y));
        const cv::Point br(cvRound(d.box.x + d.box.width), cvRound(d.box.y + d.box.height));
        cv::rectangle(out, tl, br, cv::Scalar(0, 200, 0), 2);
        putLabel(out, cv::format("%.2f", d.score), tl + cv::Point(0, -4), cv::Scalar(0, 200, 0));
    }
    return out;
}

cv::Mat drawDetections3D(const cv::Mat& image_bgr, const std::vector<Detection3D>& detections) {
    cv::Mat out = ensureBgr(image_bgr);
    for (const Detection3D& d : detections) {
        const cv::Scalar color = d.valid ? cv::Scalar(0, 180, 255) : cv::Scalar(130, 130, 130);
        const cv::Point tl(cvRound(d.box.x), cvRound(d.box.y));
        const cv::Point br(cvRound(d.box.x + d.box.width), cvRound(d.box.y + d.box.height));
        cv::rectangle(out, tl, br, color, 2);
        const std::string label =
            d.valid ? std::string(cv::format("Z=%.2f m", d.depth)) : std::string("Z=?");
        putLabel(out, label, tl + cv::Point(0, -4), color);
    }
    return out;
}

}  // namespace s3m

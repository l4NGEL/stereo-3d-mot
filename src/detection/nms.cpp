#include "s3m/detection/nms.hpp"

#include <algorithm>

namespace s3m {

float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    const float x1 = std::max(a.x, b.x);
    const float y1 = std::max(a.y, b.y);
    const float x2 = std::min(a.x + a.width, b.x + b.width);
    const float y2 = std::min(a.y + a.height, b.y + b.height);
    const float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    const float uni = a.width * a.height + b.width * b.height - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

std::vector<int> nms(const std::vector<cv::Rect2f>& boxes, const std::vector<float>& scores,
                     float iou_threshold, float score_threshold, int top_k) {
    CV_Assert(boxes.size() == scores.size());

    std::vector<int> order;
    order.reserve(boxes.size());
    for (int i = 0; i < static_cast<int>(boxes.size()); ++i) {
        if (scores[static_cast<std::size_t>(i)] >= score_threshold)
            order.push_back(i);
    }
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return scores[static_cast<std::size_t>(a)] > scores[static_cast<std::size_t>(b)];
    });
    if (top_k > 0 && static_cast<int>(order.size()) > top_k) {
        order.resize(static_cast<std::size_t>(top_k));
    }

    std::vector<int> keep;
    std::vector<char> suppressed(order.size(), 0);
    for (std::size_t i = 0; i < order.size(); ++i) {
        if (suppressed[i])
            continue;
        keep.push_back(order[i]);
        for (std::size_t j = i + 1; j < order.size(); ++j) {
            if (!suppressed[j] && iou(boxes[static_cast<std::size_t>(order[i])],
                                      boxes[static_cast<std::size_t>(order[j])]) > iou_threshold) {
                suppressed[j] = 1;
            }
        }
    }
    return keep;
}

std::vector<int> nmsClassAware(const std::vector<cv::Rect2f>& boxes,
                               const std::vector<float>& scores, const std::vector<int>& class_ids,
                               float iou_threshold, float score_threshold, int top_k) {
    CV_Assert(boxes.size() == scores.size() && boxes.size() == class_ids.size());

    // Offset every box into a per-class "lane" so that boxes of different classes
    // can never overlap, then a single NMS pass is class-aware for free.
    float max_coord = 1.0f;
    for (const cv::Rect2f& b : boxes) {
        max_coord = std::max(max_coord, std::max(b.x + b.width, b.y + b.height));
    }
    const float lane = max_coord + 1.0f;

    std::vector<cv::Rect2f> shifted(boxes.size());
    for (std::size_t i = 0; i < boxes.size(); ++i) {
        const float off = static_cast<float>(class_ids[i]) * lane;
        shifted[i] =
            cv::Rect2f(boxes[i].x + off, boxes[i].y + off, boxes[i].width, boxes[i].height);
    }
    return nms(shifted, scores, iou_threshold, score_threshold, top_k);
}

}  // namespace s3m

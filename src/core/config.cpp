#include "s3m/core/config.hpp"

#include <stdexcept>
#include <string>

#include <opencv2/core.hpp>

namespace s3m {
namespace {

template <typename T>
void readScalar(const cv::FileNode& parent, const char* key, T& out) {
    if (parent.isNone() || parent.empty()) return;
    const cv::FileNode node = parent[key];
    if (node.isNone() || node.empty()) return;
    T value{};
    node >> value;
    out = value;
}

void readBool(const cv::FileNode& parent, const char* key, bool& out) {
    if (parent.isNone() || parent.empty()) return;
    const cv::FileNode node = parent[key];
    if (node.isNone() || node.empty()) return;
    int value = out ? 1 : 0;
    node >> value;
    out = value != 0;
}

void readDoubleVector(const cv::FileNode& parent, const char* key, std::vector<double>& out) {
    if (parent.isNone() || parent.empty()) return;
    const cv::FileNode node = parent[key];
    if (node.isNone() || node.empty()) return;
    std::vector<double> values;
    node >> values;
    if (!values.empty()) out = values;
}

void readIntVector(const cv::FileNode& parent, const char* key, std::vector<int>& out) {
    if (parent.isNone() || parent.empty()) return;
    const cv::FileNode node = parent[key];
    if (node.isNone() || node.empty() || !node.isSeq()) return;
    std::vector<int> values;
    node >> values;
    out = values;  // an explicit list (even empty) is a valid override
}

}  // namespace

Config Config::load(const std::string& path) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        throw std::runtime_error("Config::load: cannot open '" + path + "'");
    }

    Config cfg;

    const cv::FileNode sm = fs["stereo_matcher"];
    readScalar(sm, "type", cfg.stereo_matcher.type);
    readScalar(sm, "min_disparity", cfg.stereo_matcher.min_disparity);
    readScalar(sm, "num_disparities", cfg.stereo_matcher.num_disparities);
    readScalar(sm, "block_size", cfg.stereo_matcher.block_size);
    readScalar(sm, "uniqueness_ratio", cfg.stereo_matcher.uniqueness_ratio);
    readScalar(sm, "speckle_window_size", cfg.stereo_matcher.speckle_window_size);
    readScalar(sm, "speckle_range", cfg.stereo_matcher.speckle_range);
    readScalar(sm, "disp12_max_diff", cfg.stereo_matcher.disp12_max_diff);
    readScalar(sm, "pre_filter_cap", cfg.stereo_matcher.pre_filter_cap);
    readScalar(sm, "p1_multiplier", cfg.stereo_matcher.p1_multiplier);
    readScalar(sm, "p2_multiplier", cfg.stereo_matcher.p2_multiplier);
    readBool(sm, "mode_hh", cfg.stereo_matcher.mode_hh);

    const cv::FileNode de = fs["depth_eval"];
    readScalar(de, "min_depth", cfg.depth_eval.min_depth);
    readScalar(de, "max_depth", cfg.depth_eval.max_depth);
    readDoubleVector(de, "bad_thresholds", cfg.depth_eval.bad_thresholds);

    const cv::FileNode det = fs["detector"];
    readScalar(det, "type", cfg.detector.type);
    readScalar(det, "model_path", cfg.detector.model_path);
    readScalar(det, "score_threshold", cfg.detector.score_threshold);
    readScalar(det, "nms_iou", cfg.detector.nms_iou);
    readScalar(det, "input_size", cfg.detector.input_size);
    readIntVector(det, "keep_classes", cfg.detector.keep_classes);

    const cv::FileNode tr = fs["tracking"];
    readScalar(tr, "dt", cfg.tracking.dt);
    readScalar(tr, "process_noise", cfg.tracking.process_noise);
    readScalar(tr, "measurement_noise", cfg.tracking.measurement_noise);
    readScalar(tr, "max_age", cfg.tracking.max_age);
    readScalar(tr, "min_hits", cfg.tracking.min_hits);
    readScalar(tr, "gating_distance", cfg.tracking.gating_distance);

    return cfg;
}

}  // namespace s3m

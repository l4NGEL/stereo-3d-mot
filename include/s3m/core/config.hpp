#pragma once

#include <string>
#include <vector>

namespace s3m {

/// Parameters for the block/semi-global stereo matcher. See configs/default.yaml.
struct StereoMatcherParams {
    std::string type = "SGBM";  ///< "BM" or "SGBM"
    int min_disparity = 0;
    int num_disparities = 128;  ///< must be a positive multiple of 16
    int block_size = 5;         ///< odd; 5..21 for BM, 3..11 for SGBM
    int uniqueness_ratio = 10;
    int speckle_window_size = 100;
    int speckle_range = 2;
    int disp12_max_diff = 1;
    int pre_filter_cap = 63;
    int p1_multiplier = 8;   ///< SGBM P1 = p1_multiplier * channels * block_size^2
    int p2_multiplier = 32;  ///< SGBM P2 = p2_multiplier * channels * block_size^2
    bool mode_hh = false;    ///< SGBM: use the full-scale two-pass DP (MODE_HH)
};

/// Parameters for depth/disparity evaluation against ground truth.
struct DepthEvalParams {
    double min_depth = 0.1;
    double max_depth = 100.0;
    std::vector<double> bad_thresholds{1.0, 2.0, 4.0};  ///< disparity-px error thresholds
};

/// Parameters for the object detector (see apps: --detector / --model).
struct DetectorParams {
    std::string type = "none";        ///< "none" | "hog" | "onnx"
    std::string model_path;           ///< ONNX model file (type == "onnx")
    double score_threshold = 0.25;
    double nms_iou = 0.45;
    int input_size = 640;             ///< square network input (type == "onnx")
    std::vector<int> keep_classes;    ///< COCO ids to keep; empty -> all
};

/// Parameters for the 3D constant-velocity tracker.
struct TrackingParams {
    double dt = 0.1;                 ///< seconds between frames
    double process_noise = 1.0;      ///< acceleration std [m/s^2]
    double measurement_noise = 0.05; ///< position measurement std [m]
    int max_age = 30;                ///< frames tolerated without a match
    int min_hits = 3;                ///< matches required to confirm a track
    double gating_distance = 2.0;    ///< maximum association distance [m]
};

/// Aggregated pipeline configuration.
struct Config {
    StereoMatcherParams stereo_matcher;
    DepthEvalParams depth_eval;
    DetectorParams detector;
    TrackingParams tracking;

    /// Parse a cv::FileStorage YAML file. Unknown or missing keys keep the
    /// default value above. Throws std::runtime_error if the file cannot be read.
    static Config load(const std::string& path);

    static Config defaults() { return Config{}; }
};

}  // namespace s3m

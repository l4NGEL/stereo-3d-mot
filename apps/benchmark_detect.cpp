// Detector latency / throughput benchmark: run a detector over frames from a
// source and report warmup-excluded latency percentiles.

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include <opencv2/core.hpp>

#include "common.hpp"
#include "s3m/core/timer.hpp"

using namespace s3m;

namespace {

void printHelp() {
    std::cout << "benchmark_detect - detector latency / throughput\n\n"
                 "  --detector hog|onnx        detector (default: onnx if a model is set)\n"
                 "  --model <path.onnx>        model for --detector onnx\n"
                 "  --source synthetic|<dir>   frame source (default: synthetic)\n"
                 "  --frames N                 timed frames (default 50)\n"
                 "  --warmup N                 untimed warmup frames (default 5)\n"
                 "  --threads N                ONNX Runtime intra-op threads\n"
                 "  --config <yaml>\n";
}

double percentile(std::vector<double> v, double p) {
    if (v.empty())
        return 0.0;
    std::sort(v.begin(), v.end());
    const double idx = p / 100.0 * static_cast<double>(v.size() - 1);
    const auto lo = static_cast<std::size_t>(idx);
    if (lo + 1 >= v.size())
        return v.back();
    const double frac = idx - static_cast<double>(lo);
    return v[lo] * (1.0 - frac) + v[lo + 1] * frac;
}

}  // namespace

int main(int argc, char** argv) {
    const app::Args args(argc, argv);
    if (args.has("help")) {
        printHelp();
        return 0;
    }

    Config cfg;
    std::unique_ptr<FrameSource> source;
    std::unique_ptr<Detector> detector;
    try {
        cfg = app::loadConfig(args);
        if (cfg.detector.type == "none" && !args.has("detector") &&
            (args.has("model") || !cfg.detector.model_path.empty())) {
            cfg.detector.type = "onnx";
        }
        source = app::makeSource(args);
        detector = app::makeDetector(args, cfg);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    const int timed = std::max(1, args.getInt("frames", 50));
    const int warmup = std::max(0, args.getInt("warmup", 5));

    std::vector<cv::Mat> frames;
    while (const auto f = source->next()) {
        frames.push_back(f->left);
        if (static_cast<int>(frames.size()) >= timed + warmup)
            break;
    }
    if (frames.empty()) {
        std::cerr << "error: source produced no frames\n";
        return 1;
    }
    const auto frame_at = [&](int i) -> const cv::Mat& {
        return frames[static_cast<std::size_t>(i) % frames.size()];
    };

    std::cout << "detector = " << detector->name() << "   frames = " << timed << " (+" << warmup
              << " warmup)\n";

    for (int i = 0; i < warmup; ++i)
        (void)detector->detect(frame_at(i));

    std::vector<double> ms;
    ms.reserve(static_cast<std::size_t>(timed));
    long long total_dets = 0;
    for (int i = 0; i < timed; ++i) {
        Stopwatch sw;
        const std::vector<Detection2D> d = detector->detect(frame_at(warmup + i));
        ms.push_back(sw.elapsedMs());
        total_dets += static_cast<long long>(d.size());
    }

    const double mean = std::accumulate(ms.begin(), ms.end(), 0.0) / static_cast<double>(ms.size());
    std::cout << "\ndetections ... " << total_dets << "  ("
              << cv::format("%.2f", static_cast<double>(total_dets) / timed) << " / frame)\n"
              << "latency ms ... mean " << cv::format("%.2f", mean) << "   p50 "
              << cv::format("%.2f", percentile(ms, 50)) << "   p90 "
              << cv::format("%.2f", percentile(ms, 90)) << "   p99 "
              << cv::format("%.2f", percentile(ms, 99)) << "\n"
              << "throughput ... " << cv::format("%.1f", mean > 0.0 ? 1000.0 / mean : 0.0)
              << " FPS\n";
    return 0;
}

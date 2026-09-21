// Quantitative disparity/depth benchmark over a frame source that provides
// ground truth. Emits a per-frame log, an aggregate summary, and (optionally)
// a Markdown table for the README.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "common.hpp"
#include "s3m/core/timer.hpp"
#include "s3m/depth/depth_metrics.hpp"
#include "s3m/depth/stereo_matcher.hpp"
#include "s3m/geometry/reprojection.hpp"

namespace fs = std::filesystem;
using namespace s3m;

namespace {

double badAt(const DepthMetrics& m, double threshold) {
    for (std::size_t i = 0; i < m.bad_thresholds.size() && i < m.bad_fraction.size(); ++i) {
        if (std::abs(m.bad_thresholds[i] - threshold) < 1e-9)
            return m.bad_fraction[i];
    }
    return 0.0;
}

struct Accumulator {
    double rmse = 0.0, mae = 0.0, abs_rel = 0.0, bad2 = 0.0, delta1 = 0.0, density = 0.0;
    int count = 0;

    void add(const DepthMetrics& m) {
        rmse += m.rmse;
        mae += m.mae;
        abs_rel += m.abs_rel;
        bad2 += badAt(m, 2.0);
        delta1 += m.delta1;
        density += m.density;
        ++count;
    }

    double mean(double sum) const { return count > 0 ? sum / count : 0.0; }

    void report(std::ostream& os, const std::string& tag) const {
        if (count == 0) {
            os << tag << ": no ground truth in this source\n";
            return;
        }
        os << tag << " (mean over " << count << " frame(s)):\n"
           << "  RMSE     = " << mean(rmse) << "\n"
           << "  MAE      = " << mean(mae) << "\n"
           << "  abs-rel  = " << mean(abs_rel) << "\n"
           << "  bad-2.0  = " << 100.0 * mean(bad2) << " %\n"
           << "  delta<1.25 = " << 100.0 * mean(delta1) << " %\n"
           << "  density  = " << 100.0 * mean(density) << " %\n";
    }
};

}  // namespace

int main(int argc, char** argv) {
    const app::Args args(argc, argv);
    if (args.has("help")) {
        std::cout << "benchmark_depth - disparity/depth accuracy + timing\n\n"
                     "  --source synthetic|<middlebury-scene-dir>   (default: synthetic)\n"
                     "  --config <yaml>     matcher / evaluation parameters\n"
                     "  --matcher BM|SGBM   override the matcher type\n"
                     "  --frames N          synthetic frame count (default 30)\n"
                     "  --out <dir>         also write CSV + Markdown table\n";
        return 0;
    }

    Config cfg = app::loadConfig(args);
    if (args.has("matcher"))
        cfg.stereo_matcher.type = args.get("matcher");

    std::unique_ptr<FrameSource> source;
    try {
        source = app::makeSource(args);
    } catch (const std::exception& e) {
        std::cerr << "source error: " << e.what() << "\n";
        return 1;
    }

    StereoMatcher matcher(cfg.stereo_matcher);
    const int crop = matcher.params().num_disparities;
    const StereoRig& rig = source->rig();

    std::vector<double> match_ms;
    Accumulator disparity_acc;
    Accumulator depth_acc;

    std::cout << std::fixed << std::setprecision(3);
    int index = 0;
    while (const auto frame = source->next()) {
        Stopwatch sw;
        const cv::Mat disparity = matcher.computeDisparity(frame->left, frame->right);
        match_ms.push_back(sw.elapsedMs());

        const cv::Mat depth = disparityToDepthMap(disparity, rig);

        const bool can_crop = crop > 0 && crop < disparity.cols;
        const cv::Rect roi = can_crop ? cv::Rect(crop, 0, disparity.cols - crop, disparity.rows)
                                      : cv::Rect(0, 0, disparity.cols, disparity.rows);

        if (frame->hasGtDisparity() && frame->gt_disparity.size() == disparity.size()) {
            const DepthMetrics m = evaluate(disparity(roi), frame->gt_disparity(roi), 0.05, 1e6,
                                            cfg.depth_eval.bad_thresholds);
            disparity_acc.add(m);
            std::cout << "frame " << index << " [disp]  " << m << "\n";
        }
        if (frame->hasGtDepth() && frame->gt_depth.size() == depth.size()) {
            const DepthMetrics m =
                evaluate(depth(roi), frame->gt_depth(roi), cfg.depth_eval.min_depth,
                         cfg.depth_eval.max_depth, cfg.depth_eval.bad_thresholds);
            depth_acc.add(m);
            std::cout << "frame " << index << " [depth] " << m << "\n";
        }
        ++index;
    }

    const double total_ms = std::accumulate(match_ms.begin(), match_ms.end(), 0.0);
    const double mean_ms = match_ms.empty() ? 0.0 : total_ms / static_cast<double>(match_ms.size());

    std::cout << "\nmatcher = " << cfg.stereo_matcher.type << "   num_disparities = " << crop
              << "   block_size = " << cfg.stereo_matcher.block_size << "\n";
    std::cout << "frames  = " << index << "   match: mean " << mean_ms << " ms  ("
              << (mean_ms > 0.0 ? 1000.0 / mean_ms : 0.0) << " FPS)\n\n";
    disparity_acc.report(std::cout, "disparity (px)");
    depth_acc.report(std::cout, "depth (m)");

    if (args.has("out")) {
        const std::string dir = args.get("out");
        fs::create_directories(dir);
        std::ofstream md(dir + "/depth_benchmark.md");
        if (md) {
            md << std::fixed << std::setprecision(3);
            md << "| metric | disparity (px) | depth (m) |\n";
            md << "|---|---|---|\n";
            md << "| RMSE | " << disparity_acc.mean(disparity_acc.rmse) << " | "
               << depth_acc.mean(depth_acc.rmse) << " |\n";
            md << "| MAE | " << disparity_acc.mean(disparity_acc.mae) << " | "
               << depth_acc.mean(depth_acc.mae) << " |\n";
            md << "| abs-rel | " << disparity_acc.mean(disparity_acc.abs_rel) << " | "
               << depth_acc.mean(depth_acc.abs_rel) << " |\n";
            md << "| bad-2.0 % | " << 100.0 * disparity_acc.mean(disparity_acc.bad2) << " | "
               << 100.0 * depth_acc.mean(depth_acc.bad2) << " |\n";
            md << "| density % | " << 100.0 * disparity_acc.mean(disparity_acc.density) << " | "
               << 100.0 * depth_acc.mean(depth_acc.density) << " |\n";
            md << "| match ms | " << mean_ms << " | - |\n";
            std::cout << "\nwrote " << dir << "/depth_benchmark.md\n";
        }
    }
    return 0;
}

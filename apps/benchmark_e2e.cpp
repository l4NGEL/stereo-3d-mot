// End-to-end latency benchmark of the shipped pipeline on a KITTI tracking
// sequence (docs/evaluation/PROTOCOL.md §3.7): image decode -> stereo -> depth ->
// detector -> promoteTo3D + appearance -> tracker, one frame at a time, exactly
// as it would run online. Nothing else is in the timed loop (no ground truth,
// no scoring, no dumping), so per-frame wall times are not distorted by
// evaluation code.
//
// Per-frame rows go to a CSV; the summary (median / mean / p95 / p99 / max and
// FPS with and without image decode) is computed from the frames after the
// warm-up prefix and printed. FPS here is throughput = 1000 / mean latency of
// the strictly sequential loop; a pipelined implementation (decode of frame
// t+1 overlapping compute of t) is not what is measured.

#include <algorithm>
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

#include <opencv2/core/utility.hpp>

#include "common.hpp"
#include "s3m/core/timer.hpp"
#include "s3m/depth/stereo_matcher.hpp"
#include "s3m/geometry/reprojection.hpp"
#include "s3m/io/kitti_loader.hpp"
#include "s3m/tracking/appearance.hpp"
#include "s3m/tracking/tracker.hpp"

using namespace s3m;
namespace fs = std::filesystem;

namespace {

void printHelp() {
    std::cout << "benchmark_e2e - per-frame latency of the shipped pipeline on a KITTI sequence\n\n"
                 "  --kitti-root <dir>   calib/ image_02/ image_03/ label_02/\n"
                 "  --sequence <NNNN>    zero-padded sequence id\n"
                 "  --config <yaml>      pipeline parameters (detector threshold, tiles, ...)\n"
                 "  --model <path.onnx>  detector model (default: config)\n"
                 "  --frames N           only the first N frames (default: all)\n"
                 "  --warmup N           frames excluded from the summary (default 10)\n"
                 "  --assoc <name>       mahalanobis3d | iou2d | fused (default: config)\n"
                 "  --threads N          cv::setNumThreads(N) and ORT intra-op threads\n"
                 "  --cuda               CUDA execution provider (GPU image only)\n"
                 "  --out <file.csv>     per-frame rows\n";
}

struct Row {
    double io_ms = 0.0;
    double stereo_ms = 0.0;
    double depth_ms = 0.0;
    double detect_ms = 0.0;
    double promote_ms = 0.0;  ///< promoteTo3D + appearance descriptors
    double track_ms = 0.0;
    double total_ms = 0.0;  ///< wall time of the whole iteration, decode included
    double computeMs() const { return total_ms - io_ms; }
};

double percentile(std::vector<double> v, double p) {
    if (v.empty())
        return 0.0;
    std::sort(v.begin(), v.end());
    const double rank = p / 100.0 * static_cast<double>(v.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(rank);
    const std::size_t hi = std::min(lo + 1, v.size() - 1);
    const double frac = rank - static_cast<double>(lo);
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

double mean(const std::vector<double>& v) {
    return v.empty() ? 0.0
                     : std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

void summarise(const std::string& label, const std::vector<double>& ms) {
    const double m = mean(ms);
    std::cout << std::left << std::setw(24) << label << " median " << percentile(ms, 50)
              << "  mean " << m << "  p95 " << percentile(ms, 95) << "  p99 " << percentile(ms, 99)
              << "  max " << percentile(ms, 100) << " ms   -> " << (m > 0.0 ? 1000.0 / m : 0.0)
              << " FPS (1000/mean), "
              << (percentile(ms, 50) > 0.0 ? 1000.0 / percentile(ms, 50) : 0.0)
              << " FPS (1000/median)\n";
}

}  // namespace

int main(int argc, char** argv) {
    const app::Args args(argc, argv);
    if (args.has("help") || !args.has("kitti-root") || !args.has("sequence")) {
        printHelp();
        return args.has("help") ? 0 : 1;
    }
    if (args.has("threads"))
        cv::setNumThreads(args.getInt("threads", -1));

    Config cfg;
    std::unique_ptr<Detector> detector;
    std::unique_ptr<KittiTrackingSource> source;
    try {
        cfg = app::loadConfig(args);
        if (cfg.detector.type == "none" && !args.has("detector"))
            cfg.detector.type = "onnx";
        detector = app::makeDetector(args, cfg);
        source =
            std::make_unique<KittiTrackingSource>(args.get("kitti-root"), args.get("sequence"));
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    TrackerParams params = TrackerParams::fromConfig(cfg.tracking);
    if (args.has("assoc")) {
        const std::string a = args.get("assoc");
        params.association = a == "iou2d"   ? AssociationMethod::kIou2D
                             : a == "fused" ? AssociationMethod::kFusedAppearance
                                            : AssociationMethod::kMahalanobis3D;
    }

    const int max_frames = args.getInt("frames", -1);
    const int warmup = args.getInt("warmup", 10);
    StereoMatcher matcher(cfg.stereo_matcher);
    Tracker tracker(params);
    const StereoRig& rig = source->rig();

    std::vector<Row> rows;
    while (true) {
        Row r;
        Stopwatch loop;
        Stopwatch sw;
        const auto frame = source->next();
        r.io_ms = sw.elapsedMs();
        if (!frame || (max_frames > 0 && static_cast<int>(rows.size()) >= max_frames))
            break;

        sw.reset();
        const cv::Mat disparity = matcher.computeDisparity(frame->left, frame->right);
        r.stereo_ms = sw.elapsedMs();

        sw.reset();
        const cv::Mat depth = disparityToDepthMap(disparity, rig);
        r.depth_ms = sw.elapsedMs();

        sw.reset();
        const std::vector<Detection2D> dets2d = detector->detect(frame->left);
        r.detect_ms = sw.elapsedMs();

        sw.reset();
        std::vector<Detection3D> dets3d = promoteTo3D(dets2d, depth, rig);
        attachAppearance(dets3d, frame->left);
        r.promote_ms = sw.elapsedMs();

        sw.reset();
        tracker.update(dets3d);
        r.track_ms = sw.elapsedMs();

        r.total_ms = loop.elapsedMs();
        rows.push_back(r);
    }

    if (args.has("out")) {
        const fs::path out(args.get("out"));
        if (out.has_parent_path())
            fs::create_directories(out.parent_path());
        std::ofstream f(out);
        f << "frame,warm,io_ms,stereo_ms,depth_ms,detect_ms,promote_ms,track_ms,compute_ms,"
             "total_ms\n"
          << std::fixed << std::setprecision(4);
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const Row& r = rows[i];
            f << i << ',' << (static_cast<int>(i) < warmup ? 1 : 0) << ',' << r.io_ms << ','
              << r.stereo_ms << ',' << r.depth_ms << ',' << r.detect_ms << ',' << r.promote_ms
              << ',' << r.track_ms << ',' << r.computeMs() << ',' << r.total_ms << '\n';
        }
    }

    std::vector<double> total, compute, io, stereo, detect, track;
    for (std::size_t i = static_cast<std::size_t>(std::max(0, warmup)); i < rows.size(); ++i) {
        total.push_back(rows[i].total_ms);
        compute.push_back(rows[i].computeMs());
        io.push_back(rows[i].io_ms);
        stereo.push_back(rows[i].stereo_ms);
        detect.push_back(rows[i].detect_ms);
        track.push_back(rows[i].track_ms);
    }

    std::cout << std::fixed << std::setprecision(2) << "sequence " << args.get("sequence") << ", "
              << rows.size() << " frames (" << total.size() << " after " << warmup
              << " warm-up), detector=" << detector->name()
              << ", cv threads=" << cv::getNumThreads()
              << ", tiles=" << cfg.stereo_matcher.num_tiles << "\n";
    summarise("total (with decode)", total);
    summarise("compute (no decode)", compute);
    summarise("decode (2x imread)", io);
    summarise("stereo", stereo);
    summarise("detect", detect);
    summarise("track", track);
    return 0;
}

// End-to-end stereo depth pipeline:
//   frame source -> disparity -> depth map -> 3D reprojection -> visualisation.
// Prints per-frame timing and, when the source provides ground truth, accuracy.

#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "common.hpp"
#include "s3m/core/timer.hpp"
#include "s3m/depth/depth_metrics.hpp"
#include "s3m/depth/stereo_matcher.hpp"
#include "s3m/detection/coco.hpp"
#include "s3m/geometry/reprojection.hpp"
#include "s3m/io/point_cloud.hpp"
#include "s3m/viz/depth_viz.hpp"

namespace fs = std::filesystem;
using namespace s3m;

namespace {

void printHelp() {
    std::cout <<
        "stereo_depth_demo - end-to-end stereo depth pipeline\n\n"
        "  --source synthetic|<middlebury-scene-dir>   (default: synthetic)\n"
        "  --config <yaml>     matcher / evaluation / detector parameters\n"
        "  --frames N          synthetic frame count (default 30)\n"
        "  --detector none|hog|onnx     object detector (default: none)\n"
        "  --model <path.onnx>          model for --detector onnx\n"
        "  --out <dir>         write the visualisation board (and clouds) here\n"
        "  --cloud             also export a PLY point cloud per frame\n"
        "  --max-depth M       depth colour-map clamp in metres (default 15)\n";
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
        source = app::makeSource(args);
        detector = app::makeDetector(args, cfg);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    const std::string out_dir = args.get("out", "");
    const bool want_cloud = args.has("cloud");
    const float max_depth = static_cast<float>(args.getDouble("max-depth", 15.0));
    if (!out_dir.empty()) fs::create_directories(out_dir);

    StereoMatcher matcher(cfg.stereo_matcher);
    const int crop = matcher.params().num_disparities;
    const StereoRig& rig = source->rig();
    const float max_disp = static_cast<float>(matcher.params().num_disparities);

    std::cout << "rig: fx=" << rig.left().fx() << "  baseline=" << rig.baseline()
              << " m  doffs=" << rig.doffs() << "\n"
              << "detector: " << detector->name() << "\n";

    ProfileRegistry prof;
    FpsMeter fps;
    int frame_count = 0;

    while (const auto frame = source->next()) {
        Stopwatch total;

        cv::Mat disparity;
        cv::Mat depth;
        cv::Mat valid;
        cv::Mat cloud;
        {
            const auto s = prof.scope("match");
            disparity = matcher.computeDisparity(frame->left, frame->right);
        }
        {
            const auto s = prof.scope("depth");
            depth = disparityToDepthMap(disparity, rig);
        }
        {
            const auto s = prof.scope("reproject");
            cloud = reproject(disparity, rig, &valid);
        }

        std::vector<Detection2D> dets2d;
        {
            const auto s = prof.scope("detect");
            dets2d = detector->detect(frame->left);
        }
        const std::vector<Detection3D> dets3d = promoteTo3D(dets2d, depth, rig);

        const double frame_ms = total.elapsedMs();
        const double now_fps = fps.tick();
        ++frame_count;

        cv::Mat left_panel =
            dets3d.empty() ? frame->left : drawDetections3D(frame->left, dets3d);
        std::vector<cv::Mat> panels{left_panel, colorizeDisparity(disparity, max_disp),
                                    colorizeDepth(depth, 0.0f, max_depth)};
        if (frame->hasGtDisparity()) {
            panels.push_back(colorizeDisparity(frame->gt_disparity, max_disp));
        }
        cv::Mat board = tile(panels, static_cast<int>(panels.size()));
        cv::putText(board, cv::format("frame %lld   %.1f ms   %.1f FPS",
                                      static_cast<long long>(frame->index), frame_ms, now_fps),
                    cv::Point(12, 24), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2,
                    cv::LINE_AA);

        const bool can_crop = crop > 0 && crop < disparity.cols;
        const cv::Rect roi = can_crop ? cv::Rect(crop, 0, disparity.cols - crop, disparity.rows)
                                      : cv::Rect(0, 0, disparity.cols, disparity.rows);

        if (frame->hasGtDisparity() && frame->gt_disparity.size() == disparity.size()) {
            std::cout << "  frame " << frame->index << " [disp]  "
                      << evaluate(disparity(roi), frame->gt_disparity(roi), 0.1, 1e6,
                                  cfg.depth_eval.bad_thresholds)
                      << "\n";
        }
        if (frame->hasGtDepth() && frame->gt_depth.size() == depth.size()) {
            std::cout << "  frame " << frame->index << " [depth] "
                      << evaluate(depth(roi), frame->gt_depth(roi), cfg.depth_eval.min_depth,
                                  cfg.depth_eval.max_depth, cfg.depth_eval.bad_thresholds)
                      << "\n";
        }
        for (const Detection3D& d : dets3d) {
            std::cout << "  frame " << frame->index << " [det]   " << cocoClassName(d.class_id)
                      << " (" << d.class_id << ")  score=" << cv::format("%.2f", d.score);
            if (d.valid) {
                std::cout << "  Z=" << cv::format("%.2f", d.depth) << " m  pos=("
                          << cv::format("%.2f, %.2f, %.2f", d.position.x, d.position.y,
                                        d.position.z)
                          << ")";
            } else {
                std::cout << "  Z=n/a";
            }
            std::cout << "\n";
        }

        if (!out_dir.empty()) {
            cv::imwrite(cv::format("%s/frame_%04lld.png", out_dir.c_str(),
                                   static_cast<long long>(frame->index)),
                        board);
            if (want_cloud) {
                writePointCloudPly(cv::format("%s/cloud_%04lld.ply", out_dir.c_str(),
                                              static_cast<long long>(frame->index)),
                                   cloud, frame->left, valid, /*stride=*/2);
            }
        }
    }

    std::cout << "\nprocessed " << frame_count << " frame(s)\n" << prof.summary();
    if (!out_dir.empty()) std::cout << "output written to " << out_dir << "/\n";
    return 0;
}

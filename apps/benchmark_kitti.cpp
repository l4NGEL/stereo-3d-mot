// 2D IoU vs 3D Mahalanobis vs fused (3D+IoU+appearance) association, on a
// real KITTI tracking sequence: runs the *actual* pipeline (stereo depth ->
// detector -> promoteTo3D -> Tracker) and scores each method against the
// sequence's ground truth with CLEAR-MOT + IDF1 (MotAccumulator) -- the
// real-data counterpart to benchmark_track's hermetic synthetic scene. See
// docs/roadmap.md and scripts/download_kitti.sh: this needs the KITTI
// tracking benchmark's images, labels and calibration downloaded locally;
// nothing here works without that.
//
// Stereo depth + detection are the expensive stages and are identical across
// every association variant compared here, so they run exactly once per
// frame (collectFrames()) regardless of how many methods/weight
// combinations --sweep asks for; only the cheap track/associate/score loop
// (run()) repeats per variant.

#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "common.hpp"
#include "s3m/core/timer.hpp"
#include "s3m/depth/stereo_matcher.hpp"
#include "s3m/geometry/reprojection.hpp"
#include "s3m/io/kitti_loader.hpp"
#include "s3m/tracking/appearance.hpp"
#include "s3m/tracking/mot_metrics.hpp"
#include "s3m/tracking/tracker.hpp"

using namespace s3m;

namespace {

void printHelp() {
    std::cout <<
        "benchmark_kitti - 2D IoU vs 3D Mahalanobis vs fused association on a real\n"
        "                  KITTI tracking sequence\n\n"
        "  --kitti-root <dir>   directory holding calib/ image_02/ image_03/ label_02/\n"
        "  --sequence <NNNN>    zero-padded sequence id, e.g. 0000 (default 0000)\n"
        "  --detector hog|onnx  object detector (default onnx)\n"
        "  --model <path.onnx>  model for --detector onnx\n"
        "  --frames N           limit to the first N frames (default: whole sequence)\n"
        "  --gate M             MOT evaluation match distance in metres (default 2.0)\n"
        "  --config <yaml>      matcher / detector / tracker parameters\n"
        "  --w3d/--wiou/--wapp  override the fused row's weights (default from config)\n"
        "  --sweep              also run a weight-sensitivity sweep and a cue ablation\n"
        "                       (3D-only / 3D+IoU / 3D+appearance / full fused)\n\n"
        "Needs the dataset downloaded first -- see scripts/download_kitti.sh.\n";
}

MotObject toMotObject(int id, const cv::Point3f& p) {
    MotObject o;
    o.id = id;
    o.position = p;
    return o;
}

std::vector<MotObject> groundTruthObjects(const std::vector<KittiObject>& kitti_objects) {
    std::vector<MotObject> out;
    out.reserve(kitti_objects.size());
    for (const KittiObject& o : kitti_objects) {
        const cv::Point3d c = o.centroid();
        out.push_back(toMotObject(
            o.track_id, cv::Point3f(static_cast<float>(c.x), static_cast<float>(c.y),
                                    static_cast<float>(c.z))));
    }
    return out;
}

/// Everything the (cheap, repeated-per-variant) tracking stage needs from one
/// frame -- computed once by collectFrames() regardless of how many
/// association variants run() is later called with.
struct FrameData {
    std::vector<Detection3D> dets3d;
    std::vector<MotObject> gt;
};

std::vector<FrameData> collectFrames(KittiTrackingSource& source, Detector& detector,
                                     const StereoMatcherParams& mp, int max_frames) {
    source.reset();
    StereoMatcher matcher(mp);
    std::vector<FrameData> frames;

    int frame_index = 0;
    while (const auto frame = source.next()) {
        if (max_frames > 0 && frame_index >= max_frames) break;

        const cv::Mat disparity = matcher.computeDisparity(frame->left, frame->right);
        const cv::Mat depth = disparityToDepthMap(disparity, source.rig());
        const std::vector<Detection2D> dets2d = detector.detect(frame->left);
        std::vector<Detection3D> dets3d = promoteTo3D(dets2d, depth, source.rig());
        attachAppearance(dets3d, frame->left);

        FrameData fd;
        fd.dets3d = std::move(dets3d);
        fd.gt = groundTruthObjects(source.objectsAt(frame_index));
        frames.push_back(std::move(fd));
        ++frame_index;
    }
    return frames;
}

/// Run one association method/weight setting over pre-computed frame data,
/// returning its MOT summary.
MotSummary run(const std::vector<FrameData>& frames, const TrackerParams& params, double gate) {
    Tracker tracker(params);
    MotAccumulator acc(gate);
    for (const FrameData& fd : frames) {
        const std::vector<TrackState> tracks = tracker.update(fd.dets3d).tracks;
        std::vector<MotObject> hyp;
        hyp.reserve(tracks.size());
        for (const TrackState& t : tracks) {
            if (t.confirmed) hyp.push_back(toMotObject(t.id, t.position));
        }
        acc.update(fd.gt, hyp);
    }
    return acc.summary();
}

void printRow(const std::string& name, const MotSummary& s) {
    std::cout << std::left << std::setw(28) << name << ": " << s.toString() << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    const app::Args args(argc, argv);
    if (args.has("help") || !args.has("kitti-root")) {
        printHelp();
        return args.has("help") ? 0 : 1;
    }

    Config cfg;
    std::unique_ptr<Detector> detector;
    std::unique_ptr<KittiTrackingSource> source;
    try {
        cfg = app::loadConfig(args);
        if (cfg.detector.type == "none" && !args.has("detector")) cfg.detector.type = "onnx";
        detector = app::makeDetector(args, cfg);
        source = std::make_unique<KittiTrackingSource>(args.get("kitti-root", ""),
                                                        args.get("sequence", "0000"));
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    if (!source->hasGroundTruth()) {
        std::cerr << "error: sequence '" << args.get("sequence", "0000")
                  << "' has no label_02 file (this looks like a test-split sequence -- "
                     "evaluation needs ground truth)\n";
        return 1;
    }

    const int max_frames = args.getInt("frames", -1);
    const double gate = args.getDouble("gate", 2.0);
    TrackerParams base_params = TrackerParams::fromConfig(cfg.tracking);
    base_params.fused_weight_3d = args.getDouble("w3d", base_params.fused_weight_3d);
    base_params.fused_weight_iou = args.getDouble("wiou", base_params.fused_weight_iou);
    base_params.fused_weight_appearance = args.getDouble("wapp", base_params.fused_weight_appearance);

    std::cout << "sequence " << args.get("sequence", "0000") << "   detector=" << detector->name()
              << "   frames=" << (max_frames > 0 ? std::to_string(max_frames) : "all")
              << "   gate=" << gate << " m\n\n";

    const std::vector<FrameData> frames = collectFrames(*source, *detector, cfg.stereo_matcher, max_frames);
    std::cout << frames.size() << " frames processed (stereo+detection run once, shared by every"
                                  " association variant below)\n\n";

    TrackerParams mahalanobis = base_params;
    mahalanobis.association = AssociationMethod::kMahalanobis3D;
    TrackerParams iou = base_params;
    iou.association = AssociationMethod::kIou2D;
    TrackerParams fused = base_params;
    fused.association = AssociationMethod::kFusedAppearance;

    printRow("3D Mahalanobis", run(frames, mahalanobis, gate));
    printRow("2D IoU", run(frames, iou, gate));
    printRow(cv::format("3D+IoU+ReID (%.1f/%.1f/%.1f)", fused.fused_weight_3d, fused.fused_weight_iou,
                        fused.fused_weight_appearance),
            run(frames, fused, gate));

    if (!args.has("sweep")) return 0;

    std::cout << "\n-- weight sensitivity (fused_weight_3d / _iou / _appearance) --\n";
    const double weight_sets[][3] = {
        {0.4, 0.3, 0.3}, {0.5, 0.2, 0.3}, {0.6, 0.1, 0.3}, {0.4, 0.2, 0.4}, {0.6, 0.2, 0.2},
    };
    for (const auto& w : weight_sets) {
        TrackerParams p = base_params;
        p.association = AssociationMethod::kFusedAppearance;
        p.fused_weight_3d = w[0];
        p.fused_weight_iou = w[1];
        p.fused_weight_appearance = w[2];
        printRow(cv::format("fused (%.1f/%.1f/%.1f)", w[0], w[1], w[2]), run(frames, p, gate));
    }

    // Cue ablation via the fused code path's weights. Caveat, stated here
    // rather than glossed over: kFusedAppearance's *gate* is always the union
    // of the Mahalanobis and IoU gates regardless of weight (tracker.cpp), so
    // zeroing a weight mutes that cue's contribution to the *cost* but not
    // its contribution to *feasibility* -- "3D+appearance, no IoU" below
    // still benefits from IoU admitting candidates Mahalanobis alone
    // wouldn't. A fully independent ablation would need separate gating per
    // combination; this is the honest, cheaper version of that experiment.
    std::cout << "\n-- cue ablation (same union gate throughout -- see comment in source for the"
                 " caveat this implies) --\n";
    printRow("3D only (plain kMahalanobis3D)", run(frames, mahalanobis, gate));
    TrackerParams iou_3d = base_params;
    iou_3d.association = AssociationMethod::kFusedAppearance;
    iou_3d.fused_weight_3d = 0.6;
    iou_3d.fused_weight_iou = 0.4;
    iou_3d.fused_weight_appearance = 0.0;
    printRow("3D+IoU, no appearance", run(frames, iou_3d, gate));
    TrackerParams app_3d = base_params;
    app_3d.association = AssociationMethod::kFusedAppearance;
    app_3d.fused_weight_3d = 0.7;
    app_3d.fused_weight_iou = 0.0;
    app_3d.fused_weight_appearance = 0.3;
    printRow("3D+appearance, no IoU", run(frames, app_3d, gate));
    printRow(cv::format("3D+IoU+appearance (%.1f/%.1f/%.1f)", fused.fused_weight_3d,
                        fused.fused_weight_iou, fused.fused_weight_appearance),
            run(frames, fused, gate));

    return 0;
}

// 2D IoU vs 3D Mahalanobis association, on a real KITTI tracking sequence:
// runs the *actual* pipeline (stereo depth -> detector -> promoteTo3D ->
// Tracker) and scores each method against the sequence's ground truth with
// CLEAR-MOT + IDF1 (MotAccumulator) -- the real-data counterpart to
// benchmark_track's hermetic synthetic scene. See docs/roadmap.md and
// scripts/download_kitti.sh: this needs the KITTI tracking benchmark's
// images, labels and calibration downloaded locally; nothing here works
// without that.

#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "common.hpp"
#include "s3m/core/timer.hpp"
#include "s3m/depth/stereo_matcher.hpp"
#include "s3m/geometry/reprojection.hpp"
#include "s3m/io/kitti_loader.hpp"
#include "s3m/tracking/mot_metrics.hpp"
#include "s3m/tracking/tracker.hpp"

using namespace s3m;

namespace {

void printHelp() {
    std::cout <<
        "benchmark_kitti - 2D IoU vs 3D Mahalanobis on a real KITTI tracking sequence\n\n"
        "  --kitti-root <dir>   directory holding calib/ image_02/ image_03/ label_02/\n"
        "  --sequence <NNNN>    zero-padded sequence id, e.g. 0000 (default 0000)\n"
        "  --detector hog|onnx  object detector (default onnx)\n"
        "  --model <path.onnx>  model for --detector onnx\n"
        "  --frames N           limit to the first N frames (default: whole sequence)\n"
        "  --gate M             MOT evaluation match distance in metres (default 2.0)\n"
        "  --config <yaml>      matcher / detector / tracker parameters\n\n"
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

/// Run one association method over the whole sequence, returning its MOT
/// summary. Re-runs the depth + detection stages every time so the two
/// methods see byte-identical input -- only the association differs.
MotSummary run(KittiTrackingSource& source, Detector& detector, const StereoMatcherParams& mp,
              AssociationMethod method, const TrackerParams& base_params, double gate,
              int max_frames) {
    source.reset();
    StereoMatcher matcher(mp);
    TrackerParams params = base_params;
    params.association = method;
    Tracker tracker(params);
    MotAccumulator acc(gate);

    int frame_index = 0;
    while (const auto frame = source.next()) {
        if (max_frames > 0 && frame_index >= max_frames) break;

        const cv::Mat disparity = matcher.computeDisparity(frame->left, frame->right);
        const cv::Mat depth = disparityToDepthMap(disparity, source.rig());
        const std::vector<Detection2D> dets2d = detector.detect(frame->left);
        const std::vector<Detection3D> dets3d = promoteTo3D(dets2d, depth, source.rig());

        // Only confirmed tracks are submitted as predictions -- tentative
        // (not yet past min_hits) tracks would inflate false positives with
        // one-frame noise, same as a real submission would exclude them.
        const std::vector<TrackState> tracks = tracker.update(dets3d).tracks;
        std::vector<MotObject> hyp;
        hyp.reserve(tracks.size());
        for (const TrackState& t : tracks) {
            if (t.confirmed) hyp.push_back(toMotObject(t.id, t.position));
        }

        acc.update(groundTruthObjects(source.objectsAt(frame_index)), hyp);
        ++frame_index;
    }
    return acc.summary();
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
    const TrackerParams base_params = TrackerParams::fromConfig(cfg.tracking);

    std::cout << "sequence " << args.get("sequence", "0000") << "   detector=" << detector->name()
              << "   frames=" << (max_frames > 0 ? std::to_string(max_frames) : "all")
              << "   gate=" << gate << " m\n\n";

    std::cout << "3D Mahalanobis: "
              << run(*source, *detector, cfg.stereo_matcher, AssociationMethod::kMahalanobis3D,
                     base_params, gate, max_frames)
                     .toString()
              << "\n";
    std::cout << "2D IoU        : "
              << run(*source, *detector, cfg.stereo_matcher, AssociationMethod::kIou2D,
                     base_params, gate, max_frames)
                     .toString()
              << "\n";
    return 0;
}

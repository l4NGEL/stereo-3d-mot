// Replays MOTChallenge public detections (det/det.txt) through the project's
// Tracker, for the outside-data check of docs/evaluation/IDSW_PLAN.md round 5.
//
// MOT17 is monocular: there is no depth, so only association rules that score
// pairs by 2D IoU are meaningful here (iou2d with or without the predicted
// box, two_stage with lost_cost=iou or none, reconfirm, filtered box). Each
// detection enters the tracker with valid = true and a placeholder 3D position
// (box centre in pixels, z = 1) that none of those rules read -- a variant that
// does read 3D (mahalanobis3d, fused, range_gated, lost_cost=app_mahal) is
// refused rather than silently evaluated on fake depth.
//
// Output: <out>/tracks_<variant>.txt in eval_kitti's dump format
//   frame id cls x y w h X Y Z vx vy vz confirmed tsu hits age
// with 0-based frames; scoring lives in scripts/eval/run_mot17_eval.py.

#include <algorithm>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.hpp"
#include "s3m/tracking/tracker.hpp"
#include "variants.hpp"

using namespace s3m;
namespace fs = std::filesystem;

namespace {

void printHelp() {
    std::cout << "eval_mot2d - replay MOTChallenge det.txt through the Tracker (2D rules only)\n"
                 "  --det <det.txt>      MOTChallenge detections (frame,id,x,y,w,h,conf,...)\n"
                 "  --frames N           sequence length (seqinfo.ini seqLength)\n"
                 "  --variants <file>    tracker variants, eval_kitti syntax\n"
                 "  --config <yaml>      base tracker parameters (default configs/default.yaml)\n"
                 "  --out <dir>          writes <dir>/tracks_<variant>.txt\n";
}

std::vector<std::vector<Detection3D>> readMotDetections(const std::string& path, int frames) {
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("cannot open '" + path + "'");
    std::vector<std::vector<Detection3D>> out(static_cast<std::size_t>(frames));
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty())
            continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream is(line);
        int frame = 0, id = 0;
        float x = 0, y = 0, w = 0, h = 0, conf = 0;
        if (!(is >> frame >> id >> x >> y >> w >> h >> conf))
            throw std::runtime_error("bad line in '" + path + "': " + line);
        if (frame < 1 || frame > frames || w <= 0.0f || h <= 0.0f)
            continue;
        Detection3D d;
        d.box = cv::Rect2f(x, y, w, h);
        d.score = conf;
        d.class_id = 0;  // person
        d.valid = true;
        d.position = cv::Point3f(x + w / 2.0f, y + h / 2.0f, 1.0f);  // placeholder, never read
        d.depth = 1.0f;
        out[static_cast<std::size_t>(frame - 1)].push_back(d);
    }
    return out;
}

void requireTwoDimensional(const app::Variant& v) {
    const TrackerParams& p = v.params;
    const bool ok = p.association == AssociationMethod::kIou2D ||
                    (p.association == AssociationMethod::kTwoStage &&
                     p.lost_cost != LostTrackCost::kAppearanceMahalanobis);
    if (!ok)
        throw std::runtime_error("variant '" + v.name +
                                 "' reads depth or appearance; not evaluable on MOT17");
}

}  // namespace

int main(int argc, char** argv) {
    const app::Args args(argc, argv);
    if (args.has("help") || !args.has("det") || !args.has("frames") || !args.has("variants") ||
        !args.has("out")) {
        printHelp();
        return args.has("help") ? 0 : 1;
    }
    try {
        const Config cfg = app::loadConfig(args);
        const std::vector<app::Variant> variants =
            app::loadVariants(args.get("variants"), TrackerParams::fromConfig(cfg.tracking), 0.0f);
        const int frames = args.getInt("frames", 0);
        const auto dets = readMotDetections(args.get("det"), frames);
        const std::string dir = args.get("out");
        fs::create_directories(dir);
        for (const app::Variant& v : variants) {
            requireTwoDimensional(v);
            Tracker tracker(v.params);
            std::ofstream f(dir + "/tracks_" + v.name + ".txt");
            f << "# frame id cls x y w h X Y Z vx vy vz confirmed tsu hits age\n"
              << std::fixed << std::setprecision(3);
            for (std::size_t i = 0; i < dets.size(); ++i) {
                std::vector<Detection3D> usable;
                for (const Detection3D& d : dets[i])
                    if (d.score >= v.min_score)
                        usable.push_back(d);
                const TrackerUpdateResult r = tracker.update(usable);
                for (const TrackState& t : r.tracks) {
                    f << i << ' ' << t.id << ' ' << t.class_id << ' ' << t.box.x << ' ' << t.box.y
                      << ' ' << t.box.width << ' ' << t.box.height << " 0 0 0 0 0 0 "
                      << (t.confirmed ? 1 : 0) << ' ' << t.time_since_update << ' ' << t.hits << ' '
                      << t.age << '\n';
                }
            }
        }
        std::cout << args.get("det") << ": " << frames << " frames, " << variants.size()
                  << " variant(s) -> " << dir << "\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

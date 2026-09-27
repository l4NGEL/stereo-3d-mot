// Evaluation harness front end for one KITTI tracking sequence (see
// docs/evaluation/PROTOCOL.md). It runs the real pipeline -- stereo depth ->
// detector -> promoteTo3D -> appearance -> Tracker -- but scores nothing itself.
// Everything a scorer needs is written to disk and evaluated by the
// independent Python code in scripts/eval/, so the metric implementation is not
// the same code that produced the tracks it is judging.
//
// The expensive stage (stereo + detector) runs once per sequence with a LOW
// detector score floor; the tracker is then replayed over the cached detections
// once per variant (a variant may raise the score threshold). This is exact,
// not an approximation: class-aware greedy NMS keeps a box only if no
// higher-scored kept box overlaps it, so NMS at floor 0.01 followed by a
// score >= t filter selects the same boxes as NMS run at threshold t.
//
// Outputs under <out>/<sequence>/:
//   meta.txt            key=value: rig intrinsics, image size, config, thread counts
//   dets.txt            every detection with score >= floor:
//                       frame cls score x y w h valid X Y Z depth
//   timing.csv          per-frame stage times [ms]
//   tracks_<variant>.txt  every tracker state after each frame (confirmed and
//                       tentative, coasting included -- the scorer applies the
//                       emission policy):
//                       frame id cls x y w h X Y Z vx vy vz confirmed tsu hits age
//   track_timing.csv    variant,frame,track_ms

#include <algorithm>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
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
    std::cout << "eval_kitti - dump detections, timings and tracker output of one KITTI tracking\n"
                 "             sequence for the independent scorer in scripts/eval/\n\n"
                 "  --kitti-root <dir>   calib/ image_02/ image_03/ label_02/\n"
                 "  --sequence <NNNN>    zero-padded sequence id\n"
                 "  --out <dir>          output root (writes <dir>/<sequence>/...)\n"
                 "  --config <yaml>      matcher / detector / tracker parameters\n"
                 "  --model <path.onnx>  detector model (default: config)\n"
                 "  --frames N           only the first N frames (default: all)\n"
                 "  --score-floor S      detector score floor for the dump (default 0.01)\n"
                 "  --variants <file>    tracker variants, one per line: name key=value ...\n"
                 "                       keys: assoc score min_hits max_age w3d wiou wapp\n"
                 "                       hungarian meas_noise proc_noise gate_chi2 iou_gate\n"
                 "                       range_gate (range_gated association only, metres)\n"
                 "                       (default: the three as-shipped associations)\n"
                 "  --threads N          cv::setNumThreads(N) and ORT intra-op threads\n"
                 "  --cuda               CUDA execution provider (GPU image only)\n"
                 "  --tag <text>         free-form label stored in meta.txt (e.g. git hash)\n";
}

struct Variant {
    std::string name;
    TrackerParams params;
    float min_score = 0.25f;
};

struct FrameDets {
    std::vector<Detection3D> dets;
};

struct FrameTiming {
    double capture_ms = 0.0;
    double stereo_ms = 0.0;
    double depth_ms = 0.0;
    double detect_ms = 0.0;
    double promote_ms = 0.0;
    double appearance_ms = 0.0;
};

AssociationMethod parseAssociation(const std::string& s) {
    if (s == "mahalanobis3d")
        return AssociationMethod::kMahalanobis3D;
    if (s == "iou2d")
        return AssociationMethod::kIou2D;
    if (s == "fused")
        return AssociationMethod::kFusedAppearance;
    if (s == "range_gated")
        return AssociationMethod::kRangeGated3D;
    throw std::runtime_error("unknown association '" + s + "'");
}

Variant parseVariantLine(const std::string& line, const TrackerParams& base, float base_score) {
    std::istringstream is(line);
    Variant v;
    v.params = base;
    v.min_score = base_score;
    is >> v.name;
    std::string token;
    while (is >> token) {
        const std::size_t eq = token.find('=');
        if (eq == std::string::npos)
            throw std::runtime_error("variant '" + v.name + "': expected key=value, got '" + token +
                                     "'");
        const std::string key = token.substr(0, eq);
        const std::string val = token.substr(eq + 1);
        if (key == "assoc")
            v.params.association = parseAssociation(val);
        else if (key == "score")
            v.min_score = std::stof(val);
        else if (key == "min_hits")
            v.params.min_hits = std::stoi(val);
        else if (key == "max_age")
            v.params.max_age = std::stoi(val);
        else if (key == "w3d")
            v.params.fused_weight_3d = std::stod(val);
        else if (key == "wiou")
            v.params.fused_weight_iou = std::stod(val);
        else if (key == "wapp")
            v.params.fused_weight_appearance = std::stod(val);
        else if (key == "hungarian")
            v.params.use_hungarian = std::stoi(val) != 0;
        else if (key == "meas_noise")
            v.params.measurement_noise = std::stod(val);
        else if (key == "proc_noise")
            v.params.process_noise = std::stod(val);
        else if (key == "gate_chi2")
            v.params.gating_chi2 = std::stod(val);
        else if (key == "iou_gate")
            v.params.iou_gate = std::stod(val);
        else if (key == "range_gate")
            v.params.range_gate_max_m = std::stod(val);
        else
            throw std::runtime_error("variant '" + v.name + "': unknown key '" + key + "'");
    }
    return v;
}

std::vector<Variant> loadVariants(const std::string& path, const TrackerParams& base,
                                  float base_score) {
    std::vector<Variant> out;
    if (path.empty()) {
        const std::pair<const char*, AssociationMethod> shipped[] = {
            {"shipped_mahalanobis3d", AssociationMethod::kMahalanobis3D},
            {"shipped_iou2d", AssociationMethod::kIou2D},
            {"shipped_fused", AssociationMethod::kFusedAppearance},
        };
        for (const auto& s : shipped) {
            Variant v;
            v.name = s.first;
            v.params = base;
            v.params.association = s.second;
            v.min_score = base_score;
            out.push_back(v);
        }
        return out;
    }
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("cannot open variants file '" + path + "'");
    std::string line;
    while (std::getline(f, line)) {
        const std::size_t first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos || line[first] == '#')
            continue;
        out.push_back(parseVariantLine(line, base, base_score));
    }
    return out;
}

std::vector<FrameDets> runFrontend(KittiTrackingSource& source, Detector& detector,
                                   const StereoMatcherParams& mp, int max_frames,
                                   std::vector<FrameTiming>& timing) {
    source.reset();
    StereoMatcher matcher(mp);
    std::vector<FrameDets> frames;

    int frame_index = 0;
    while (true) {
        Stopwatch sw;
        const auto frame = source.next();
        const double capture_ms = sw.elapsedMs();
        if (!frame)
            break;
        if (max_frames > 0 && frame_index >= max_frames)
            break;

        FrameTiming t;
        t.capture_ms = capture_ms;

        sw.reset();
        const cv::Mat disparity = matcher.computeDisparity(frame->left, frame->right);
        t.stereo_ms = sw.elapsedMs();

        sw.reset();
        const cv::Mat depth = disparityToDepthMap(disparity, source.rig());
        t.depth_ms = sw.elapsedMs();

        sw.reset();
        const std::vector<Detection2D> dets2d = detector.detect(frame->left);
        t.detect_ms = sw.elapsedMs();

        sw.reset();
        std::vector<Detection3D> dets3d = promoteTo3D(dets2d, depth, source.rig());
        t.promote_ms = sw.elapsedMs();

        sw.reset();
        attachAppearance(dets3d, frame->left);
        t.appearance_ms = sw.elapsedMs();

        FrameDets fd;
        fd.dets = std::move(dets3d);
        frames.push_back(std::move(fd));
        timing.push_back(t);
        ++frame_index;
    }
    return frames;
}

void writeMeta(const std::string& path, const app::Args& args, const KittiTrackingSource& source,
               const Config& cfg, std::size_t frames, double floor_score) {
    std::ofstream f(path);
    const StereoRig& rig = source.rig();
    f << std::setprecision(10);
    f << "sequence=" << args.get("sequence") << "\n"
      << "frames=" << frames << "\n"
      << "image_width=" << rig.left().imageSize().width << "\n"
      << "image_height=" << rig.left().imageSize().height << "\n"
      << "fx=" << rig.left().fx() << "\n"
      << "fy=" << rig.left().fy() << "\n"
      << "cx=" << rig.left().cx() << "\n"
      << "cy=" << rig.left().cy() << "\n"
      << "baseline=" << rig.baseline() << "\n"
      << "doffs=" << rig.doffs() << "\n"
      << "config=" << args.get("config", "(default)") << "\n"
      << "model=" << args.get("model", cfg.detector.model_path) << "\n"
      << "score_floor=" << floor_score << "\n"
      << "num_tiles=" << cfg.stereo_matcher.num_tiles << "\n"
      << "cv_threads=" << cv::getNumThreads() << "\n"
      << "cuda=" << (args.has("cuda") || cfg.detector.use_cuda ? 1 : 0) << "\n"
      << "tag=" << args.get("tag", "") << "\n";
}

void writeDets(const std::string& path, const std::vector<FrameDets>& frames) {
    std::ofstream f(path);
    f << "# frame cls score x y w h valid X Y Z depth\n" << std::fixed << std::setprecision(6);
    for (std::size_t i = 0; i < frames.size(); ++i) {
        for (const Detection3D& d : frames[i].dets) {
            f << i << ' ' << d.class_id << ' ' << d.score << ' ' << d.box.x << ' ' << d.box.y << ' '
              << d.box.width << ' ' << d.box.height << ' ' << (d.valid ? 1 : 0) << ' '
              << d.position.x << ' ' << d.position.y << ' ' << d.position.z << ' ' << d.depth
              << '\n';
        }
    }
}

void writeFrontendTiming(const std::string& path, const std::vector<FrameTiming>& timing) {
    std::ofstream f(path);
    f << "frame,capture_ms,stereo_ms,depth_ms,detect_ms,promote_ms,appearance_ms\n"
      << std::fixed << std::setprecision(4);
    for (std::size_t i = 0; i < timing.size(); ++i) {
        const FrameTiming& t = timing[i];
        f << i << ',' << t.capture_ms << ',' << t.stereo_ms << ',' << t.depth_ms << ','
          << t.detect_ms << ',' << t.promote_ms << ',' << t.appearance_ms << '\n';
    }
}

void runVariant(const Variant& v, const std::vector<FrameDets>& frames, const std::string& dir,
                std::ofstream& track_timing) {
    Tracker tracker(v.params);
    std::ofstream f(dir + "/tracks_" + v.name + ".txt");
    f << "# frame id cls x y w h X Y Z vx vy vz confirmed tsu hits age\n"
      << std::fixed << std::setprecision(6);
    for (std::size_t i = 0; i < frames.size(); ++i) {
        std::vector<Detection3D> usable;
        usable.reserve(frames[i].dets.size());
        for (const Detection3D& d : frames[i].dets) {
            if (d.score >= v.min_score)
                usable.push_back(d);
        }
        Stopwatch sw;
        const TrackerUpdateResult r = tracker.update(usable);
        track_timing << v.name << ',' << i << ',' << sw.elapsedMs() << '\n';
        for (const TrackState& t : r.tracks) {
            f << i << ' ' << t.id << ' ' << t.class_id << ' ' << t.box.x << ' ' << t.box.y << ' '
              << t.box.width << ' ' << t.box.height << ' ' << t.position.x << ' ' << t.position.y
              << ' ' << t.position.z << ' ' << t.velocity.x << ' ' << t.velocity.y << ' '
              << t.velocity.z << ' ' << (t.confirmed ? 1 : 0) << ' ' << t.time_since_update << ' '
              << t.hits << ' ' << t.age << '\n';
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    const app::Args args(argc, argv);
    if (args.has("help") || !args.has("kitti-root") || !args.has("sequence") || !args.has("out")) {
        printHelp();
        return args.has("help") ? 0 : 1;
    }
    if (args.has("threads"))
        cv::setNumThreads(args.getInt("threads", -1));

    Config cfg;
    std::unique_ptr<Detector> detector;
    std::unique_ptr<KittiTrackingSource> source;
    std::vector<Variant> variants;
    const double floor_score = args.getDouble("score-floor", 0.01);
    try {
        cfg = app::loadConfig(args);
        const float shipped_score = static_cast<float>(cfg.detector.score_threshold);
        if (cfg.detector.type == "none" && !args.has("detector"))
            cfg.detector.type = "onnx";
        cfg.detector.score_threshold = floor_score;
        detector = app::makeDetector(args, cfg);
        source =
            std::make_unique<KittiTrackingSource>(args.get("kitti-root"), args.get("sequence"));
        variants = loadVariants(args.get("variants", ""), TrackerParams::fromConfig(cfg.tracking),
                                shipped_score);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    const std::string dir = args.get("out") + "/" + args.get("sequence");
    fs::create_directories(dir);

    std::vector<FrameTiming> timing;
    const std::vector<FrameDets> frames =
        runFrontend(*source, *detector, cfg.stereo_matcher, args.getInt("frames", -1), timing);
    writeMeta(dir + "/meta.txt", args, *source, cfg, frames.size(), floor_score);
    writeDets(dir + "/dets.txt", frames);
    writeFrontendTiming(dir + "/timing.csv", timing);

    std::ofstream track_timing(dir + "/track_timing.csv");
    track_timing << "variant,frame,track_ms\n" << std::fixed << std::setprecision(4);
    for (const Variant& v : variants)
        runVariant(v, frames, dir, track_timing);

    std::cout << "sequence " << args.get("sequence") << ": " << frames.size() << " frames, "
              << variants.size() << " variant(s) -> " << dir << "\n";
    return 0;
}

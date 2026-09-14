// 2D-IoU vs 3D-Mahalanobis data association, head to head, on a hermetic
// synthetic scene with exact ground-truth identity and geometry -- no dataset
// download required.
//
// Reports both this project's own identity-preservation metrics (ID
// switches, ID consistency, fragmentation, false-track rate) AND standard
// MOTA/MOTP/IDF1 (via MotAccumulator, scored against the scene's exact,
// noise-free ground truth). This is still a controlled synthetic proxy, NOT
// the real-data evaluation -- see benchmark_kitti for that -- but the
// MOTA/MOTP/IDF1 numbers here are the same accumulator, same formulas,
// exercised end to end before trusting it on real data. It exists to make one
// specific, checkable claim: does gating association on metric depth reduce
// identity errors versus gating on 2D box overlap alone, when detections are
// noisy and objects can overlap on screen while sitting at different depths.
//
// The scene is two cards on a collision course: one near (2 m) drifting right,
// one far (9 m) drifting left, so their 2D boxes substantially overlap for many
// frames around the midpoint even though they're never close in 3D.

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "common.hpp"
#include "s3m/camera/camera_model.hpp"
#include "s3m/core/timer.hpp"
#include "s3m/io/synthetic_source.hpp"
#include "s3m/tracking/appearance.hpp"
#include "s3m/tracking/mot_metrics.hpp"
#include "s3m/tracking/tracker.hpp"

using namespace s3m;

namespace {

void printHelp() {
    std::cout <<
        "benchmark_track - 2D IoU vs 3D Mahalanobis association, head to head\n\n"
        "  --frames N           frames to simulate (default 120)\n"
        "  --drift PX           per-frame horizontal drift of each card (default 3.0)\n"
        "  --box-jitter PX      Gaussian std on simulated box position/size (default 3.0)\n"
        "  --depth-jitter M     Gaussian std on simulated depth (default 0.15)\n"
        "  --miss-prob P        chance a real card goes undetected this frame (default 0.05)\n"
        "  --fp-rate P          chance of a spurious false-positive box this frame (default 0.03)\n"
        "  --meas-noise M       tracker's assumed position noise [m] (default 2x --depth-jitter)\n"
        "  --eval-gate M        MOT evaluation match distance in metres (default 1.5)\n"
        "  --seed N             RNG seed (default 7)\n"
        "  --config <yaml>      base tracker parameters (dt, noise, gates, ...)\n";
}

/// One simulated detection plus which ground-truth card it came from (-1 for a
/// synthetic false positive with no real identity).
struct SimDetection {
    Detection3D det;
    int gt_card = -1;
};

/// Fraction of card `i`'s box covered by a strictly nearer card's box this
/// frame. The renderer already paints nearer cards over farther ones (painter's
/// algorithm), so a detector realistically stops firing on a mostly-covered
/// box -- this is what makes the crossing scene an actual occlusion, not just
/// two boxes that happen to overlap while staying independently visible.
double occludedFraction(std::size_t i, const std::vector<cv::Rect>& boxes,
                        const std::vector<SyntheticStereoSource::Card>& cards) {
    const cv::Rect& bi = boxes[i];
    if (bi.area() <= 0) return 0.0;
    double covered = 0.0;
    for (std::size_t j = 0; j < boxes.size(); ++j) {
        if (j == i || cards[j].depth_m >= cards[i].depth_m) continue;
        covered = std::max(covered, static_cast<double>((bi & boxes[j]).area()));
    }
    return covered / static_cast<double>(bi.area());
}

std::vector<SimDetection> simulateDetections(const SyntheticStereoSource& src, int frame_index,
                                             cv::RNG& rng, double box_jitter_px,
                                             double depth_jitter_m, double miss_prob,
                                             double fp_rate) {
    std::vector<SimDetection> out;
    const std::vector<cv::Rect> boxes = src.cardBoxes(frame_index);
    const std::vector<SyntheticStereoSource::Card>& cards = src.options().cards;
    const CameraModel& cam = src.rig().left();

    for (std::size_t i = 0; i < boxes.size(); ++i) {
        if (rng.uniform(0.0, 1.0) < miss_prob) continue;
        if (occludedFraction(i, boxes, cards) > 0.6) continue;  // realistically invisible

        const cv::Rect& b = boxes[i];
        const double w = std::max(4.0, b.width + rng.gaussian(box_jitter_px));
        const double h = std::max(4.0, b.height + rng.gaussian(box_jitter_px));
        const cv::Rect2f box(static_cast<float>(b.x + rng.gaussian(box_jitter_px)),
                             static_cast<float>(b.y + rng.gaussian(box_jitter_px)),
                             static_cast<float>(w), static_cast<float>(h));
        const double depth = std::max(0.05, cards[i].depth_m + rng.gaussian(depth_jitter_m));
        const cv::Point2d center(box.x + box.width * 0.5, box.y + box.height * 0.5);
        const cv::Point3d p3 = cam.backProject(center, depth);

        Detection3D d;
        d.box = box;
        d.position = cv::Point3f(static_cast<float>(p3.x), static_cast<float>(p3.y),
                                 static_cast<float>(p3.z));
        d.depth = static_cast<float>(depth);
        d.score = 0.9f;
        d.class_id = 0;
        d.valid = true;
        out.push_back(SimDetection{d, static_cast<int>(i)});
    }

    if (rng.uniform(0.0, 1.0) < fp_rate) {
        const cv::Size sz = cam.imageSize();
        const cv::Rect2f box(static_cast<float>(rng.uniform(0, std::max(1, sz.width - 40))),
                             static_cast<float>(rng.uniform(0, std::max(1, sz.height - 40))), 40.0f,
                             40.0f);
        const double depth = rng.uniform(1.0, 20.0);
        const cv::Point2d center(box.x + box.width * 0.5, box.y + box.height * 0.5);
        const cv::Point3d p3 = cam.backProject(center, depth);

        Detection3D d;
        d.box = box;
        d.position = cv::Point3f(static_cast<float>(p3.x), static_cast<float>(p3.y),
                                 static_cast<float>(p3.z));
        d.depth = static_cast<float>(depth);
        d.score = 0.35f;
        d.class_id = 0;
        d.valid = true;
        out.push_back(SimDetection{d, -1});
    }
    return out;
}

/// The scene's *exact*, noise-free ground truth for one frame -- separate
/// from simulateDetections()'s noisy/lossy view of it, exactly like a real
/// dataset's annotations are independent of whatever a detector produces.
std::vector<MotObject> cleanGtObjects(const SyntheticStereoSource& src, int frame_index) {
    const std::vector<cv::Rect> boxes = src.cardBoxes(frame_index);
    const std::vector<SyntheticStereoSource::Card>& cards = src.options().cards;
    const CameraModel& cam = src.rig().left();

    std::vector<MotObject> out;
    out.reserve(boxes.size());
    for (std::size_t i = 0; i < boxes.size(); ++i) {
        const cv::Rect& b = boxes[i];
        const cv::Point2d center(b.x + b.width * 0.5, b.y + b.height * 0.5);
        const cv::Point3d p = cam.backProject(center, cards[i].depth_m);
        MotObject o;
        o.id = static_cast<int>(i);
        o.position =
            cv::Point3f(static_cast<float>(p.x), static_cast<float>(p.y), static_cast<float>(p.z));
        out.push_back(o);
    }
    return out;
}

struct MethodResult {
    std::string name;
    int id_switches = 0;
    double id_consistency_pct = 0.0;  ///< mean over GT cards
    int fragmentation = 0;            ///< sum over GT cards of (distinct ids seen - 1)
    double false_track_rate_pct = 0.0;
    int tracks_born = 0;
    double mean_update_us = 0.0;
    MotSummary mot;  ///< standard CLEAR-MOT + IDF1, scored against exact ground truth
};

MethodResult run(const SyntheticStereoSource::Options& scene_opts, const TrackerParams& params,
                 const std::string& name, double box_jitter_px, double depth_jitter_m,
                 double miss_prob, double fp_rate, std::uint64_t seed, double eval_gate) {
    SyntheticStereoSource src(scene_opts);
    cv::RNG rng(static_cast<std::uint64_t>(seed));
    Tracker tracker(params);
    MotAccumulator mot_acc(eval_gate);

    std::vector<std::vector<std::pair<int, int>>> card_history(scene_opts.cards.size());
    std::map<int, bool> track_matched_real;
    std::vector<double> update_us;
    update_us.reserve(static_cast<std::size_t>(scene_opts.num_frames));

    for (int f = 0; f < scene_opts.num_frames; ++f) {
        const auto frame = src.next();  // same cursor order as cardBoxes(f) below -- see SyntheticStereoSource::next()
        const std::vector<SimDetection> sims =
            simulateDetections(src, f, rng, box_jitter_px, depth_jitter_m, miss_prob, fp_rate);
        std::vector<Detection3D> dets;
        dets.reserve(sims.size());
        for (const SimDetection& s : sims) dets.push_back(s.det);
        if (frame) attachAppearance(dets, frame->left);

        Stopwatch sw;
        const TrackerUpdateResult res = tracker.update(dets);
        update_us.push_back(sw.elapsedMs() * 1000.0);

        std::vector<MotObject> hyp;
        for (const TrackState& t : res.tracks) {
            if (!t.confirmed) continue;
            MotObject o;
            o.id = t.id;
            o.position = t.position;
            hyp.push_back(o);
        }
        mot_acc.update(cleanGtObjects(src, f), hyp);

        for (std::size_t k = 0; k < sims.size(); ++k) {
            const int tid = res.detection_track_id[k];
            if (tid < 0) continue;
            if (sims[k].gt_card >= 0) {
                card_history[static_cast<std::size_t>(sims[k].gt_card)].emplace_back(f, tid);
                track_matched_real[tid] = true;
            } else if (track_matched_real.find(tid) == track_matched_real.end()) {
                track_matched_real[tid] = false;
            }
        }
    }

    MethodResult r;
    r.name = name;
    double consistency_sum = 0.0;
    int consistency_n = 0;
    for (const auto& hist : card_history) {
        if (hist.empty()) continue;
        for (std::size_t i = 1; i < hist.size(); ++i) {
            if (hist[i].second != hist[i - 1].second) ++r.id_switches;
        }
        std::map<int, int> counts;
        for (const auto& frame_id : hist) ++counts[frame_id.second];
        int best = 0;
        for (const auto& id_count : counts) best = std::max(best, id_count.second);
        consistency_sum += 100.0 * static_cast<double>(best) / static_cast<double>(hist.size());
        ++consistency_n;
        r.fragmentation += static_cast<int>(counts.size()) - 1;
    }
    r.id_consistency_pct = consistency_n > 0 ? consistency_sum / consistency_n : 0.0;

    r.tracks_born = static_cast<int>(track_matched_real.size());
    int fake = 0;
    for (const auto& id_matched : track_matched_real) {
        if (!id_matched.second) ++fake;
    }
    r.false_track_rate_pct = r.tracks_born > 0 ? 100.0 * fake / r.tracks_born : 0.0;
    r.mean_update_us =
        update_us.empty() ? 0.0 : std::accumulate(update_us.begin(), update_us.end(), 0.0) /
                                       static_cast<double>(update_us.size());
    r.mot = mot_acc.summary();
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    const app::Args args(argc, argv);
    if (args.has("help")) {
        printHelp();
        return 0;
    }

    Config cfg;
    try {
        cfg = app::loadConfig(args);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    TrackerParams base = TrackerParams::fromConfig(cfg.tracking);

    SyntheticStereoSource::Options scene;
    scene.num_frames = args.getInt("frames", 120);
    scene.card_drift_px = args.getDouble("drift", 3.0);
    // Two cards on a collision course: near (2 m, drifts right -- even index)
    // crosses far (9 m, drifts left -- odd index) around the midpoint, so their
    // 2D boxes overlap substantially while their depths never do.
    scene.cards = {
        {cv::Rect(150, 180, 90, 140), 2.0},
        {cv::Rect(420, 180, 110, 140), 9.0},
    };

    const double box_jitter = args.getDouble("box-jitter", 3.0);
    const double depth_jitter = args.getDouble("depth-jitter", 0.15);
    const double miss_prob = args.getDouble("miss-prob", 0.05);
    const double fp_rate = args.getDouble("fp-rate", 0.03);
    const double eval_gate = args.getDouble("eval-gate", 1.5);
    const auto seed = static_cast<std::uint64_t>(args.getInt("seed", 7));

    // The Mahalanobis gate is only as good as the tracker's belief about its
    // own measurement noise. Stereo depth error grows with Z^2 (at 9 m here, a
    // single pixel of disparity noise is already ~1.3 m), so defaulting to
    // configs/default.yaml's close-range measurement_noise (0.05 m) would gate
    // out plenty of *correct* matches, not just wrong ones -- that's a
    // mistuned filter, not an association-quality difference.
    //
    // Setting measurement_noise to exactly the simulated depth-jitter std
    // isn't enough margin either: a chi-square(3, 95%) gate rejects ~5% of
    // genuinely-correct matches *by construction* when R is tuned right at the
    // true noise level, and each false rejection here briefly spawns a second
    // track that flip-flops with the original for a few frames before dying --
    // one rare event inflates the ID-switch count a lot. Real systems pad R
    // for exactly this reason. Default --meas-noise to 2x depth-jitter.
    base.measurement_noise = args.getDouble("meas-noise", depth_jitter * 2.0);

    std::cout << "scene: " << scene.num_frames << " frames, drift=" << scene.card_drift_px
              << " px/frame, cards at 2.0 m (->) and 9.0 m (<-); the near card occludes"
                 " (and the far card's detections drop out) while they cross on screen\n"
              << "noise: box_jitter=" << box_jitter << " px  depth_jitter=" << depth_jitter
              << " m  miss_prob=" << miss_prob << "  fp_rate=" << fp_rate
              << "\ntracker: measurement_noise=" << base.measurement_noise
              << " m (set from depth_jitter; override with --meas-noise)\n\n";

    TrackerParams mahalanobis = base;
    mahalanobis.association = AssociationMethod::kMahalanobis3D;
    mahalanobis.use_hungarian = true;

    TrackerParams iou = base;
    iou.association = AssociationMethod::kIou2D;
    iou.use_hungarian = true;

    TrackerParams fused = base;
    fused.association = AssociationMethod::kFusedAppearance;
    fused.use_hungarian = true;

    TrackerParams mahalanobis_greedy = mahalanobis;
    mahalanobis_greedy.use_hungarian = false;
    TrackerParams iou_greedy = iou;
    iou_greedy.use_hungarian = false;

    const std::vector<MethodResult> results = {
        run(scene, mahalanobis, "3D Mahalanobis + Hungarian", box_jitter, depth_jitter, miss_prob,
            fp_rate, seed, eval_gate),
        run(scene, iou, "2D IoU        + Hungarian", box_jitter, depth_jitter, miss_prob, fp_rate,
            seed, eval_gate),
        run(scene, fused, "3D+IoU+ReID    + Hungarian", box_jitter, depth_jitter, miss_prob,
            fp_rate, seed, eval_gate),
        run(scene, mahalanobis_greedy, "3D Mahalanobis + Greedy   ", box_jitter, depth_jitter,
            miss_prob, fp_rate, seed, eval_gate),
        run(scene, iou_greedy, "2D IoU        + Greedy   ", box_jitter, depth_jitter, miss_prob,
            fp_rate, seed, eval_gate),
    };

    std::cout << "-- this project's identity-preservation metrics --\n"
              << cv::format("%-28s %10s %14s %8s %10s %10s %14s\n", "method", "ID switch",
                            "ID consist.%", "fragm.", "false trk%", "tracks", "update (us)");
    for (const MethodResult& r : results) {
        std::cout << cv::format("%-28s %10d %13.1f%% %8d %9.1f%% %10d %14.2f\n", r.name.c_str(),
                                r.id_switches, r.id_consistency_pct, r.fragmentation,
                                r.false_track_rate_pct, r.tracks_born, r.mean_update_us);
    }
    std::cout << "\n-- CLEAR-MOT + IDF1 (MotAccumulator, gate=" << eval_gate << " m) --\n"
              << cv::format("%-28s %8s %8s %8s %8s %8s\n", "method", "MOTA", "MOTP", "IDF1", "IDSW",
                            "Frag");
    for (const MethodResult& r : results) {
        std::cout << cv::format("%-28s %8.3f %8.3f %8.3f %8lld %8lld\n", r.name.c_str(), r.mot.mota,
                                r.mot.motp, r.mot.idf1,
                                static_cast<long long>(r.mot.num_switches),
                                static_cast<long long>(r.mot.num_fragmentations));
    }
    std::cout << "\nlower ID switches / fragmentation / false-track% / MOTP and higher ID "
                 "consistency% / MOTA / IDF1 are better.\n"
                 "still a hermetic synthetic proxy -- benchmark_kitti runs the same "
                 "MotAccumulator on real detections and real ground truth.\n";
    return 0;
}

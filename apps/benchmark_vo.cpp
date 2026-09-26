// Phase 7: runs VisualOdometry over a real KITTI odometry sequence and
// scores the estimated trajectory against the benchmark's real ego-motion
// ground truth with ATE/RPE (trajectory_eval.hpp) -- the real-data
// counterpart to the Python synthetic-motion validation the pose/scale math
// itself was checked against before being ported to C++ (see
// docs/roadmap.md Phase 7). Needs the odometry benchmark's data downloaded
// locally first -- see scripts/fetch_kitti_odometry.py. This is a different
// KITTI benchmark from the one benchmark_kitti uses (tracking): only the
// odometry benchmark ships ego-motion ground truth.

#include <algorithm>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "common.hpp"
#include "s3m/io/kitti_odometry_loader.hpp"
#include "s3m/vo/trajectory_eval.hpp"
#include "s3m/vo/visual_odometry.hpp"

using namespace s3m;

namespace {

void printHelp() {
    std::cout
        << "benchmark_vo - stereo visual odometry on a real KITTI odometry sequence\n\n"
           "  --kitti-root <dir>   directory holding sequences/ and poses/ (odometry\n"
           "                       benchmark layout -- see scripts/fetch_kitti_odometry.py)\n"
           "  --sequence <NN>      zero-padded sequence id, e.g. 04 (default 04)\n"
           "  --frames N           limit to the first N frames (default: whole sequence)\n"
           "  --rpe-delta N        RPE relative-step size in frames (default 1)\n"
           "  --config <yaml>      stereo_matcher parameters (num_tiles, etc.)\n"
           "  --out <dir>          also write trajectory_est.csv / trajectory_gt.csv\n"
           "                       (frame,x,y,z, world frame), poses_est.txt / poses_gt.txt\n"
           "                       (KITTI 12-number camera-to-world rows, full rotation, for\n"
           "                       the independent scorer in scripts/eval/) and frames.csv\n"
           "                       (per-frame ok / matches / inliers / scale)\n\n"
           "Needs the dataset downloaded first -- see scripts/fetch_kitti_odometry.py.\n";
}

void writeCsv(const std::string& path, const std::vector<Pose>& poses) {
    std::ofstream f(path);
    f << "frame,x,y,z\n";
    for (std::size_t i = 0; i < poses.size(); ++i) {
        f << i << ',' << poses[i].position.x << ',' << poses[i].position.y << ','
          << poses[i].position.z << '\n';
    }
}

/// KITTI odometry pose format: one line per frame, the row-major 3x4 [R | t]
/// camera-to-world matrix -- readable by scripts/eval/vo_eval.py and by the
/// official devkit.
void writeKittiPoses(const std::string& path, const std::vector<Pose>& poses) {
    std::ofstream f(path);
    f << std::scientific << std::setprecision(9);
    for (const Pose& p : poses) {
        const cv::Matx33d& R = p.rotation;
        f << R(0, 0) << ' ' << R(0, 1) << ' ' << R(0, 2) << ' ' << p.position.x << ' ' << R(1, 0)
          << ' ' << R(1, 1) << ' ' << R(1, 2) << ' ' << p.position.y << ' ' << R(2, 0) << ' '
          << R(2, 1) << ' ' << R(2, 2) << ' ' << p.position.z << '\n';
    }
}

void writeFrameLog(const std::string& path, const std::vector<VoFrameResult>& log) {
    std::ofstream f(path);
    f << "frame,ok,matches,inliers,scale\n" << std::fixed << std::setprecision(6);
    for (std::size_t i = 0; i < log.size(); ++i) {
        f << i << ',' << (log[i].ok ? 1 : 0) << ',' << log[i].num_matches << ','
          << log[i].num_inliers << ',' << log[i].scale << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    const app::Args args(argc, argv);
    if (args.has("help") || !args.has("kitti-root")) {
        printHelp();
        return args.has("help") ? 0 : 1;
    }

    Config cfg;
    std::unique_ptr<KittiOdometrySource> source;
    try {
        cfg = app::loadConfig(args);
        source = std::make_unique<KittiOdometrySource>(args.get("kitti-root", ""),
                                                       args.get("sequence", "04"));
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    const int max_frames = args.getInt("frames", -1);
    const int rpe_delta = args.getInt("rpe-delta", 1);

    VisualOdometryParams vo_params;
    vo_params.stereo_matcher = cfg.stereo_matcher;
    VisualOdometry vo(source->rig(), vo_params);

    std::cout << "sequence " << args.get("sequence", "04")
              << "   frames=" << (max_frames > 0 ? std::to_string(max_frames) : "all")
              << "   ground_truth=" << (source->hasGroundTruth() ? "yes" : "no") << "\n\n";

    std::vector<Pose> estimated;
    std::vector<VoFrameResult> frame_log;
    int coasted = 0;
    std::vector<int> match_counts;
    std::vector<double> scales;

    int frame_index = 0;
    while (const auto frame = source->next()) {
        if (max_frames > 0 && frame_index >= max_frames)
            break;

        const VoFrameResult r = vo.processFrame(frame->left, frame->right);
        estimated.push_back(r.pose);
        frame_log.push_back(r);
        if (!r.ok && frame_index > 0)
            ++coasted;
        if (r.num_matches > 0)
            match_counts.push_back(r.num_matches);
        if (r.ok && frame_index > 0)
            scales.push_back(r.scale);

        ++frame_index;
    }

    std::cout << estimated.size() << " frames processed, " << coasted
              << " coasted (untrusted frame, held last pose)\n";
    if (!match_counts.empty()) {
        const double mean_matches = std::accumulate(match_counts.begin(), match_counts.end(), 0.0) /
                                    static_cast<double>(match_counts.size());
        std::cout << "mean ratio-test matches/frame: " << mean_matches << "\n";
    }
    if (!scales.empty()) {
        std::sort(scales.begin(), scales.end());
        std::cout << "recovered scale (median across trusted frames): " << scales[scales.size() / 2]
                  << " m/step\n";
    }

    if (source->hasGroundTruth()) {
        std::vector<Pose> gt(
            source->groundTruthPoses().begin(),
            source->groundTruthPoses().begin() + static_cast<long>(estimated.size()));
        const TrajectoryError err = evaluateTrajectory(estimated, gt, rpe_delta);
        std::cout << "\nATE RMSE:        " << err.ate_rmse << " m\n"
                  << "ATE mean:        " << err.ate_mean << " m\n"
                  << "RPE trans RMSE:  " << err.rpe_trans_rmse << " m / " << rpe_delta
                  << "-frame step (" << err.rpe_num_pairs << " pairs)\n"
                  << "RPE rot RMSE:    " << err.rpe_rot_rmse_deg << " deg / " << rpe_delta
                  << "-frame step\n";

        if (args.has("out")) {
            const std::string out = args.get("out");
            std::filesystem::create_directories(out);
            writeCsv(out + "/trajectory_est.csv", estimated);
            writeCsv(out + "/trajectory_gt.csv", gt);
            writeKittiPoses(out + "/poses_gt.txt", gt);
            std::cout << "\nwrote " << out << "/trajectory_{est,gt}.csv, poses_gt.txt\n";
        }
    } else if (args.has("out")) {
        std::filesystem::create_directories(args.get("out"));
        writeCsv(args.get("out") + "/trajectory_est.csv", estimated);
        std::cout << "\nwrote " << args.get("out")
                  << "/trajectory_est.csv (no ground truth to compare)\n";
    }

    if (args.has("out")) {
        const std::string out = args.get("out");
        std::filesystem::create_directories(out);
        writeKittiPoses(out + "/poses_est.txt", estimated);
        writeFrameLog(out + "/frames.csv", frame_log);
        std::cout << "wrote " << out << "/poses_est.txt, frames.csv\n";
    }

    return 0;
}

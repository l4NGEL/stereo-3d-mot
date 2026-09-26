// Runs the project's StereoMatcher over stereo pairs and writes the raw
// disparity maps as PFM (float32, kInvalidDisparity = -1 where there is no
// estimate), so the error metrics can be computed by an independent script
// (scripts/eval/stereo_eval.py) instead of by the code being judged, and so
// that pixels with no estimate stay visible (the project's own benchmark_depth
// only scores pixels that have one).
//
//   dump_disparity --source synthetic [--seed S] [--frames N] --out <dir>
//   dump_disparity --stereo2015 <root> [--max-pairs N] --out <dir>
//
// Synthetic mode also writes the exact ground truth (gt_*.pfm, gtdepth_*.pfm)
// and the rig in meta.txt. Stereo-2015 mode reads <root>/image_2/*_10.png and
// <root>/image_3/*_10.png (see scripts/fetch_kitti_stereo2015.py).

#include <algorithm>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "common.hpp"
#include "s3m/core/timer.hpp"
#include "s3m/depth/stereo_matcher.hpp"
#include "s3m/io/pfm.hpp"

using namespace s3m;
namespace fs = std::filesystem;

namespace {

void printHelp() {
    std::cout
        << "dump_disparity - write StereoMatcher disparity maps as PFM for external scoring\n\n"
           "  --source synthetic|<middlebury-dir>   frame source with ground truth\n"
           "  --seed S --frames N                    synthetic scene seed / frame count\n"
           "  --stereo2015 <root>                    KITTI Stereo 2015 layout instead\n"
           "  --max-pairs N                          limit the Stereo 2015 pairs\n"
           "  --config <yaml>                        matcher parameters\n"
           "  --matcher BM|SGBM  --block N  --tiles N  --num-disparities N   overrides\n"
           "  --out <dir>                            output directory\n";
}

std::string pad(int n) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%03d", n);
    return buf;
}

void writeRigMeta(const std::string& path, const StereoRig& rig, const StereoMatcherParams& mp) {
    std::ofstream f(path);
    f << std::setprecision(10) << "fx=" << rig.left().fx() << "\nfy=" << rig.left().fy()
      << "\ncx=" << rig.left().cx() << "\ncy=" << rig.left().cy() << "\nbaseline=" << rig.baseline()
      << "\ndoffs=" << rig.doffs() << "\nimage_width=" << rig.left().imageSize().width
      << "\nimage_height=" << rig.left().imageSize().height << "\ntype=" << mp.type
      << "\nnum_disparities=" << mp.num_disparities << "\nblock_size=" << mp.block_size
      << "\nnum_tiles=" << mp.num_tiles << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    const app::Args args(argc, argv);
    if (args.has("help") || !args.has("out")) {
        printHelp();
        return args.has("help") ? 0 : 1;
    }

    Config cfg;
    try {
        cfg = app::loadConfig(args);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    if (args.has("matcher"))
        cfg.stereo_matcher.type = args.get("matcher");
    if (args.has("block"))
        cfg.stereo_matcher.block_size = args.getInt("block", cfg.stereo_matcher.block_size);
    if (args.has("tiles"))
        cfg.stereo_matcher.num_tiles = args.getInt("tiles", cfg.stereo_matcher.num_tiles);
    if (args.has("num-disparities"))
        cfg.stereo_matcher.num_disparities =
            args.getInt("num-disparities", cfg.stereo_matcher.num_disparities);

    const std::string out = args.get("out");
    fs::create_directories(out);
    StereoMatcher matcher(cfg.stereo_matcher);
    std::ofstream timing(out + "/timing.csv");
    timing << "name,match_ms\n" << std::fixed << std::setprecision(4);

    try {
        if (args.has("stereo2015")) {
            const fs::path root(args.get("stereo2015"));
            std::vector<std::string> names;
            for (const fs::directory_entry& e : fs::directory_iterator(root / "image_2")) {
                const std::string n = e.path().filename().string();
                if (n.size() > 7 && n.substr(n.size() - 7) == "_10.png")
                    names.push_back(n);
            }
            std::sort(names.begin(), names.end());
            const int max_pairs = args.getInt("max-pairs", -1);
            if (max_pairs > 0 && static_cast<int>(names.size()) > max_pairs)
                names.resize(static_cast<std::size_t>(max_pairs));
            if (names.empty())
                throw std::runtime_error("no *_10.png pairs under " + root.string());

            fs::create_directories(out + "/disp_est");
            for (const std::string& n : names) {
                const cv::Mat left = cv::imread((root / "image_2" / n).string(), cv::IMREAD_COLOR);
                const cv::Mat right = cv::imread((root / "image_3" / n).string(), cv::IMREAD_COLOR);
                if (left.empty() || right.empty())
                    throw std::runtime_error("cannot read pair " + n);
                Stopwatch sw;
                const cv::Mat disparity = matcher.computeDisparity(left, right);
                timing << n << ',' << sw.elapsedMs() << '\n';
                writePfm(out + "/disp_est/" + n.substr(0, n.size() - 4) + ".pfm", disparity);
            }
            std::cout << names.size() << " Stereo 2015 pairs -> " << out << "/disp_est\n";
            return 0;
        }

        const std::unique_ptr<FrameSource> source = app::makeSource(args);
        writeRigMeta(out + "/meta.txt", source->rig(), matcher.params());
        int index = 0;
        while (const auto frame = source->next()) {
            Stopwatch sw;
            const cv::Mat disparity = matcher.computeDisparity(frame->left, frame->right);
            timing << pad(index) << ',' << sw.elapsedMs() << '\n';
            writePfm(out + "/est_" + pad(index) + ".pfm", disparity);
            if (frame->hasGtDisparity())
                writePfm(out + "/gt_" + pad(index) + ".pfm", frame->gt_disparity);
            if (frame->hasGtDepth())
                writePfm(out + "/gtdepth_" + pad(index) + ".pfm", frame->gt_depth);
            ++index;
        }
        std::cout << index << " frames -> " << out << "\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

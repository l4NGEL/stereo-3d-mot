#include "s3m/io/kitti_loader.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>

#include "s3m/camera/camera_model.hpp"
#include "s3m/camera/stereo_rig.hpp"

namespace s3m {
namespace {

namespace fs = std::filesystem;

std::string trim(const std::string& s) {
    const std::size_t begin = s.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return "";
    const std::size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(begin, end - begin + 1);
}

std::string zeroPad(int n, int width) {
    std::ostringstream os;
    os << std::setw(width) << std::setfill('0') << n;
    return os.str();
}

std::string withTrailingSlash(std::string dir) {
    if (!dir.empty() && dir.back() != '/' && dir.back() != '\\') dir += '/';
    return dir;
}

/// "Key: v0 v1 v2 ..." -> (Key without the colon, the values).
std::pair<std::string, std::vector<double>> parseCalibLine(const std::string& line) {
    std::istringstream is(line);
    std::string key;
    is >> key;
    if (!key.empty() && key.back() == ':') key.pop_back();
    std::vector<double> values;
    double v = 0.0;
    while (is >> v) values.push_back(v);
    return std::make_pair(key, values);
}

std::vector<int> listFrameIds(const std::string& image_dir) {
    std::vector<int> ids;
    if (!fs::exists(image_dir)) return ids;
    for (const fs::directory_entry& entry : fs::directory_iterator(image_dir)) {
        if (!entry.is_regular_file()) continue;
        try {
            ids.push_back(std::stoi(entry.path().stem().string()));
        } catch (const std::exception&) {
            continue;  // not a "NNNNNN.png"-style name, ignore
        }
    }
    std::sort(ids.begin(), ids.end());
    return ids;
}

}  // namespace

std::vector<KittiObject> readKittiLabels(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("readKittiLabels: cannot open '" + path + "'");

    std::vector<KittiObject> objects;
    std::string line;
    while (std::getline(file, line)) {
        if (trim(line).empty()) continue;
        std::istringstream is(line);
        KittiObject o;
        double bbox_left = 0.0, bbox_top = 0.0, bbox_right = 0.0, bbox_bottom = 0.0;
        const bool ok = static_cast<bool>(
            is >> o.frame >> o.track_id >> o.type >> o.truncated >> o.occluded >> o.alpha >>
            bbox_left >> bbox_top >> bbox_right >> bbox_bottom >> o.height >> o.width >>
            o.length >> o.location.x >> o.location.y >> o.location.z >> o.rotation_y);
        if (!ok) continue;  // malformed line -- skip it, keep parsing the rest

        o.bbox = cv::Rect2f(static_cast<float>(bbox_left), static_cast<float>(bbox_top),
                            static_cast<float>(bbox_right - bbox_left),
                            static_cast<float>(bbox_bottom - bbox_top));
        double score = 0.0;
        if (is >> score) o.score = score;  // present in result files, absent in GT

        objects.push_back(o);
    }
    return objects;
}

StereoRig readKittiCalib(const std::string& path, cv::Size image_size, const std::string& left_key,
                         const std::string& right_key) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("readKittiCalib: cannot open '" + path + "'");

    std::map<std::string, std::vector<double>> rows;
    std::string line;
    while (std::getline(file, line)) {
        if (trim(line).empty()) continue;
        const std::pair<std::string, std::vector<double>> parsed = parseCalibLine(line);
        if (!parsed.first.empty()) rows[parsed.first] = parsed.second;
    }

    const auto p2 = rows.find(left_key);
    const auto p3 = rows.find(right_key);
    if (p2 == rows.end() || p2->second.size() < 12 || p3 == rows.end() || p3->second.size() < 12) {
        throw std::runtime_error("readKittiCalib: '" + path + "' is missing " + left_key + " / " +
                                 right_key);
    }
    const std::vector<double>& P2 = p2->second;
    const std::vector<double>& P3 = p3->second;

    // Rectified 3x4 projection P = K * [I | t], t = (-baseline, 0, 0):
    //   the left 3x3 block of P *is* K, and P[0,3] = -fx * baseline_from_cam0.
    // fy and cy must be shared across a rectified pair (that's what makes the
    // epipolar lines horizontal); cx is allowed to differ -- that's exactly
    // what StereoRig::doffs() exists to carry, so it's read from P3 rather
    // than assumed equal to P2's.
    const double fx = P2[0];
    const double fy = P2[5];
    const double cx_left = P2[2];
    const double cx_right = P3[2];
    const double cy = P2[6];
    if (fx <= 0.0 || fy <= 0.0) {
        throw std::runtime_error("readKittiCalib: '" + path + "' has a non-positive focal length");
    }
    const double baseline = std::abs((P2[3] - P3[3]) / fx);
    if (!std::isfinite(baseline) || baseline <= 0.0 || baseline > 5.0) {
        throw std::runtime_error("readKittiCalib: '" + path + "' gives an implausible baseline");
    }

    const CameraModel left(fx, fy, cx_left, cy, image_size);
    const CameraModel right(fx, fy, cx_right, cy, image_size);
    StereoRig rig(left, right, baseline);
    rig.setDoffs(cx_right - cx_left);
    return rig;
}

KittiTrackingSource::KittiTrackingSource(const std::string& root, const std::string& sequence) {
    const std::string base = withTrailingSlash(root);
    image02_dir_ = base + "image_02/" + sequence + "/";
    image03_dir_ = base + "image_03/" + sequence + "/";

    frame_ids_ = listFrameIds(image02_dir_);
    if (frame_ids_.empty()) {
        throw std::runtime_error("KittiTrackingSource: no images under '" + image02_dir_ + "'");
    }

    const cv::Mat probe =
        cv::imread(image02_dir_ + zeroPad(frame_ids_.front(), 6) + ".png", cv::IMREAD_COLOR);
    if (probe.empty()) {
        throw std::runtime_error("KittiTrackingSource: cannot read the first frame under '" +
                                 image02_dir_ + "'");
    }
    rig_ = readKittiCalib(base + "calib/" + sequence + ".txt", probe.size());

    try {
        for (const KittiObject& o : readKittiLabels(base + "label_02/" + sequence + ".txt")) {
            if (o.isDontCare()) continue;
            labels_by_frame_[o.frame].push_back(o);
        }
    } catch (const std::runtime_error&) {
        // No label file: expected for the benchmark's test split.
    }
}

std::optional<StereoFrame> KittiTrackingSource::next() {
    if (cursor_ >= static_cast<int>(frame_ids_.size())) return std::nullopt;

    const int kitti_frame = frame_ids_[static_cast<std::size_t>(cursor_)];
    const std::string name = zeroPad(kitti_frame, 6) + ".png";

    StereoFrame frame;
    frame.index = cursor_;
    frame.timestamp = static_cast<double>(cursor_) / 10.0;  // KITTI tracking is ~10 Hz
    frame.left = cv::imread(image02_dir_ + name, cv::IMREAD_COLOR);
    frame.right = cv::imread(image03_dir_ + name, cv::IMREAD_COLOR);
    if (frame.left.empty() || frame.right.empty()) {
        throw std::runtime_error("KittiTrackingSource: cannot read frame '" + name + "'");
    }

    ++cursor_;
    return frame;
}

std::vector<KittiObject> KittiTrackingSource::objectsAt(int frame_index) const {
    if (frame_index < 0 || frame_index >= static_cast<int>(frame_ids_.size())) {
        return std::vector<KittiObject>();
    }
    const int kitti_frame = frame_ids_[static_cast<std::size_t>(frame_index)];
    const auto it = labels_by_frame_.find(kitti_frame);
    return it != labels_by_frame_.end() ? it->second : std::vector<KittiObject>();
}

}  // namespace s3m

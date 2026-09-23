#include "s3m/io/kitti_odometry_loader.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>

namespace s3m {
namespace {

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

int countLines(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("KittiOdometrySource: cannot open '" + path + "'");
    int n = 0;
    std::string line;
    while (std::getline(file, line)) {
        if (!trim(line).empty()) ++n;
    }
    return n;
}

}  // namespace

std::vector<Pose> readKittiPoses(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("readKittiPoses: cannot open '" + path + "'");

    std::vector<Pose> poses;
    std::string line;
    while (std::getline(file, line)) {
        if (trim(line).empty()) continue;
        std::istringstream is(line);
        double v[12];
        for (double& x : v) {
            if (!(is >> x)) {
                throw std::runtime_error("readKittiPoses: '" + path +
                                         "' has a line with fewer than 12 numbers");
            }
        }
        Pose pose;
        pose.rotation = cv::Matx33d(v[0], v[1], v[2], v[4], v[5], v[6], v[8], v[9], v[10]);
        pose.position = cv::Point3d(v[3], v[7], v[11]);
        poses.push_back(pose);
    }
    return poses;
}

StereoRig readKittiOdometryCalib(const std::string& path, cv::Size image_size) {
    return readKittiCalib(path, image_size, "P0", "P1");
}

KittiOdometrySource::KittiOdometrySource(const std::string& root, const std::string& sequence) {
    const std::string seq_dir = withTrailingSlash(root) + "sequences/" + sequence + "/";
    image0_dir_ = seq_dir + "image_0/";
    image1_dir_ = seq_dir + "image_1/";

    frame_count_ = countLines(seq_dir + "times.txt");
    if (frame_count_ <= 0) {
        throw std::runtime_error("KittiOdometrySource: '" + seq_dir + "times.txt' has no frames");
    }

    const cv::Mat probe = cv::imread(image0_dir_ + zeroPad(0, 6) + ".png", cv::IMREAD_COLOR);
    if (probe.empty()) {
        throw std::runtime_error("KittiOdometrySource: cannot read the first frame under '" +
                                 image0_dir_ + "'");
    }
    rig_ = readKittiOdometryCalib(seq_dir + "calib.txt", probe.size());

    const std::string poses_path = withTrailingSlash(root) + "poses/" + sequence + ".txt";
    std::ifstream probe_poses(poses_path);
    if (probe_poses) {
        poses_ = readKittiPoses(poses_path);
        if (static_cast<int>(poses_.size()) != frame_count_) {
            throw std::runtime_error("KittiOdometrySource: '" + poses_path + "' has " +
                                     std::to_string(poses_.size()) + " poses but sequence '" +
                                     sequence + "' has " + std::to_string(frame_count_) + " frames");
        }
    }
}

std::optional<StereoFrame> KittiOdometrySource::next() {
    if (cursor_ >= frame_count_) return std::nullopt;

    const std::string name = zeroPad(cursor_, 6) + ".png";
    StereoFrame frame;
    frame.index = cursor_;
    frame.timestamp = static_cast<double>(cursor_) / 10.0;  // KITTI odometry is ~10 Hz
    frame.left = cv::imread(image0_dir_ + name, cv::IMREAD_COLOR);
    frame.right = cv::imread(image1_dir_ + name, cv::IMREAD_COLOR);
    if (frame.left.empty() || frame.right.empty()) {
        throw std::runtime_error("KittiOdometrySource: cannot read frame '" + name + "'");
    }

    ++cursor_;
    return frame;
}

}  // namespace s3m

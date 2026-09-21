#include "s3m/io/middlebury_source.hpp"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "s3m/io/pfm.hpp"

namespace s3m {
namespace {

std::string trim(const std::string& s) {
    const std::size_t begin = s.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos)
        return "";
    const std::size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(begin, end - begin + 1);
}

/// Parse "[a b c; d e f; g h i]" into up to 9 doubles (row-major).
std::vector<double> parseMatrix(const std::string& value) {
    std::string s = value;
    for (char& c : s) {
        if (c == '[' || c == ']' || c == ';' || c == ',')
            c = ' ';
    }
    std::istringstream is(s);
    std::vector<double> values;
    double v = 0.0;
    while (is >> v)
        values.push_back(v);
    return values;
}

std::string baseName(std::string path) {
    while (!path.empty() && (path.back() == '/' || path.back() == '\\'))
        path.pop_back();
    const std::size_t pos = path.find_last_of("/\\");
    return pos == std::string::npos ? path : path.substr(pos + 1);
}

struct Calib {
    double fx0 = 0, fy0 = 0, cx0 = 0, cy0 = 0;
    double fx1 = 0, fy1 = 0, cx1 = 0, cy1 = 0;
    double doffs = 0;
    double baseline_mm = 0;
    int width = 0, height = 0;
};

Calib parseCalib(const std::string& path) {
    std::ifstream file(path);
    if (!file)
        throw std::runtime_error("MiddleburySource: cannot open '" + path + "'");

    Calib calib;
    std::string line;
    while (std::getline(file, line)) {
        const std::size_t eq = line.find('=');
        if (eq == std::string::npos)
            continue;
        const std::string key = trim(line.substr(0, eq));
        const std::string val = trim(line.substr(eq + 1));
        try {
            if (key == "cam0") {
                const std::vector<double> m = parseMatrix(val);
                if (m.size() >= 9) {
                    calib.fx0 = m[0];
                    calib.cx0 = m[2];
                    calib.fy0 = m[4];
                    calib.cy0 = m[5];
                }
            } else if (key == "cam1") {
                const std::vector<double> m = parseMatrix(val);
                if (m.size() >= 9) {
                    calib.fx1 = m[0];
                    calib.cx1 = m[2];
                    calib.fy1 = m[4];
                    calib.cy1 = m[5];
                }
            } else if (key == "doffs") {
                calib.doffs = std::stod(val);
            } else if (key == "baseline") {
                calib.baseline_mm = std::stod(val);
            } else if (key == "width") {
                calib.width = std::stoi(val);
            } else if (key == "height") {
                calib.height = std::stoi(val);
            }
        } catch (const std::exception&) {
            // ignore a malformed individual line, keep parsing the rest
        }
    }
    return calib;
}

}  // namespace

MiddleburySource::MiddleburySource(const std::string& scene_dir) {
    scene_name_ = baseName(scene_dir);

    std::string dir = scene_dir;
    if (!dir.empty() && dir.back() != '/' && dir.back() != '\\')
        dir += '/';

    const Calib calib = parseCalib(dir + "calib.txt");

    left_ = cv::imread(dir + "im0.png", cv::IMREAD_COLOR);
    right_ = cv::imread(dir + "im1.png", cv::IMREAD_COLOR);
    if (left_.empty() || right_.empty()) {
        throw std::runtime_error("MiddleburySource: missing im0.png / im1.png in '" + dir + "'");
    }

    const cv::Size size = left_.size();
    const double scale = (calib.width > 0 && calib.width != size.width)
                             ? static_cast<double>(size.width) / static_cast<double>(calib.width)
                             : 1.0;

    const double fy0 = calib.fy0 > 0 ? calib.fy0 : calib.fx0;
    const double fx1 = calib.fx1 > 0 ? calib.fx1 : calib.fx0;
    const double fy1 = calib.fy1 > 0 ? calib.fy1 : fy0;
    const double cx1 = calib.cx1 > 0 ? calib.cx1 : calib.cx0;
    const double cy1 = calib.cy1 > 0 ? calib.cy1 : calib.cy0;

    CameraModel cam_left =
        CameraModel(calib.fx0, fy0, calib.cx0, calib.cy0, size).scaled(scale, scale);
    CameraModel cam_right = CameraModel(fx1, fy1, cx1, cy1, size).scaled(scale, scale);

    const double baseline_m = (calib.baseline_mm > 0 ? calib.baseline_mm : 100.0) / 1000.0;
    rig_ = StereoRig(cam_left, cam_right, baseline_m);

    const double doffs =
        (calib.doffs != 0.0 ? calib.doffs * scale : (cam_right.cx() - cam_left.cx()));
    rig_.setDoffs(doffs);

    try {
        cv::Mat disparity = readPfm(dir + "disp0.pfm");
        if (disparity.type() == CV_32FC1)
            gt_disparity_ = disparity;
    } catch (const std::exception&) {
        // ground truth is optional
    }
}

std::optional<StereoFrame> MiddleburySource::next() {
    if (served_)
        return std::nullopt;
    served_ = true;

    StereoFrame frame;
    frame.index = 0;
    frame.timestamp = 0.0;
    frame.left = left_.clone();
    frame.right = right_.clone();

    if (!gt_disparity_.empty()) {
        frame.gt_disparity = gt_disparity_.clone();

        cv::Mat depth(gt_disparity_.size(), CV_32F, cv::Scalar::all(0));
        for (int y = 0; y < depth.rows; ++y) {
            const float* d_row = gt_disparity_.ptr<float>(y);
            float* z_row = depth.ptr<float>(y);
            for (int x = 0; x < depth.cols; ++x) {
                if (std::isfinite(d_row[x]) && d_row[x] > 0.0f) {
                    z_row[x] = static_cast<float>(rig_.disparityToDepth(d_row[x]));
                }
            }
        }
        frame.gt_depth = depth;
    }
    return frame;
}

}  // namespace s3m

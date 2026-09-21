#include "common.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "s3m/io/middlebury_source.hpp"
#include "s3m/io/synthetic_source.hpp"
#if defined(S3M_WITH_ONNX)
#include "s3m/detection/onnx_detector.hpp"
#endif

namespace s3m::app {

namespace {
bool looksLikeFlag(const char* s) {
    return s != nullptr && s[0] == '-' && s[1] == '-';
}
}  // namespace

Args::Args(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string token = argv[i];
        if (token.rfind("--", 0) != 0) {
            positional_.push_back(token);
            continue;
        }
        token = token.substr(2);
        const std::size_t eq = token.find('=');
        if (eq != std::string::npos) {
            values_[token.substr(0, eq)] = token.substr(eq + 1);
        } else if (i + 1 < argc && !looksLikeFlag(argv[i + 1])) {
            values_[token] = argv[++i];
        } else {
            values_[token] = "true";
        }
    }
}

bool Args::has(const std::string& key) const {
    return values_.find(key) != values_.end();
}

std::string Args::get(const std::string& key, const std::string& fallback) const {
    const auto it = values_.find(key);
    return it == values_.end() ? fallback : it->second;
}

int Args::getInt(const std::string& key, int fallback) const {
    const auto it = values_.find(key);
    if (it == values_.end())
        return fallback;
    try {
        return std::stoi(it->second);
    } catch (const std::exception&) {
        return fallback;
    }
}

double Args::getDouble(const std::string& key, double fallback) const {
    const auto it = values_.find(key);
    if (it == values_.end())
        return fallback;
    try {
        return std::stod(it->second);
    } catch (const std::exception&) {
        return fallback;
    }
}

Config loadConfig(const Args& args) {
    if (args.has("config")) {
        return Config::load(args.get("config"));
    }
    const char* kDefault = "configs/default.yaml";
    std::ifstream probe(kDefault);
    if (probe.good()) {
        try {
            return Config::load(kDefault);
        } catch (const std::exception&) {
            // fall through to defaults
        }
    }
    return Config::defaults();
}

std::unique_ptr<FrameSource> makeSource(const Args& args) {
    const std::string source = args.get("source", "synthetic");
    if (source.empty() || source == "synthetic") {
        SyntheticStereoSource::Options options;
        options.num_frames = args.getInt("frames", 30);
        if (args.has("seed")) {
            options.seed = static_cast<std::uint64_t>(args.getInt("seed", 42));
        }
        return std::make_unique<SyntheticStereoSource>(options);
    }
    return std::make_unique<MiddleburySource>(source);
}

std::unique_ptr<Detector> makeDetector(const Args& args, const Config& cfg) {
    const std::string type = args.get("detector", cfg.detector.type);

    if (type == "none" || type.empty()) {
        return std::make_unique<NullDetector>();
    }
    if (type == "hog") {
        return std::make_unique<HogPeopleDetector>();
    }
    if (type == "onnx") {
#if defined(S3M_WITH_ONNX)
        OnnxDetector::Options o;
        o.model_path = args.get("model", cfg.detector.model_path);
        o.input_size = args.getInt("input-size", cfg.detector.input_size);
        o.score_threshold =
            static_cast<float>(args.getDouble("score", cfg.detector.score_threshold));
        o.nms_iou = static_cast<float>(args.getDouble("nms-iou", cfg.detector.nms_iou));
        o.num_threads = args.getInt("threads", 0);
        o.keep_classes = cfg.detector.keep_classes;
        o.use_cuda = args.has("cuda") || cfg.detector.use_cuda;
        o.cuda_device_id = args.getInt("cuda-device", cfg.detector.cuda_device_id);
        return std::make_unique<OnnxDetector>(std::move(o));
#else
        std::cerr << "warning: this build has no ONNX Runtime (S3M_WITH_ONNX=OFF); "
                     "falling back to the null detector\n";
        return std::make_unique<NullDetector>();
#endif
    }

    throw std::runtime_error("makeDetector: unknown detector type '" + type + "'");
}

}  // namespace s3m::app

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "s3m/core/config.hpp"
#include "s3m/detection/detector.hpp"
#include "s3m/io/frame_source.hpp"

namespace s3m::app {

/// Minimal CLI parser: supports "--key value", "--key=value" and "--flag".
class Args {
 public:
    Args(int argc, char** argv);

    bool has(const std::string& key) const;
    std::string get(const std::string& key, const std::string& fallback = "") const;
    int getInt(const std::string& key, int fallback) const;
    double getDouble(const std::string& key, double fallback) const;
    const std::vector<std::string>& positional() const { return positional_; }

 private:
    std::map<std::string, std::string> values_;
    std::vector<std::string> positional_;
};

/// Load configuration: --config <path> if given, else configs/default.yaml if it
/// exists, else built-in defaults. Never throws for a missing default file.
Config loadConfig(const Args& args);

/// Build a frame source from --source:
///   "synthetic" (default)  -> SyntheticStereoSource (honours --frames, --seed)
///   any other value        -> MiddleburySource(<that path>)
std::unique_ptr<FrameSource> makeSource(const Args& args);

/// Build a detector from --detector (none | hog | onnx) and --model, falling
/// back to cfg.detector. Requesting "onnx" in a build without ONNX Runtime
/// prints a warning and returns a NullDetector.
std::unique_ptr<Detector> makeDetector(const Args& args, const Config& cfg);

}  // namespace s3m::app

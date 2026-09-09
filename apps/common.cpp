#include "common.hpp"

#include <cstdint>
#include <fstream>
#include <stdexcept>

#include "s3m/io/middlebury_source.hpp"
#include "s3m/io/synthetic_source.hpp"

namespace s3m::app {

namespace {
bool looksLikeFlag(const char* s) { return s != nullptr && s[0] == '-' && s[1] == '-'; }
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

bool Args::has(const std::string& key) const { return values_.find(key) != values_.end(); }

std::string Args::get(const std::string& key, const std::string& fallback) const {
    const auto it = values_.find(key);
    return it == values_.end() ? fallback : it->second;
}

int Args::getInt(const std::string& key, int fallback) const {
    const auto it = values_.find(key);
    if (it == values_.end()) return fallback;
    try {
        return std::stoi(it->second);
    } catch (const std::exception&) {
        return fallback;
    }
}

double Args::getDouble(const std::string& key, double fallback) const {
    const auto it = values_.find(key);
    if (it == values_.end()) return fallback;
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

}  // namespace s3m::app

#pragma once

#include <string>
#include <vector>

#include "s3m/tracking/tracker.hpp"

namespace s3m::app {

/// One tracker configuration replayed over cached detections (eval_kitti,
/// eval_mot2d). Variant files hold one per line: `name key=value ...`; the
/// keys and their meaning are listed in eval_kitti --help.
struct Variant {
    std::string name;
    TrackerParams params;
    float min_score = 0.25f;
};

AssociationMethod parseAssociation(const std::string& s);

/// Parse one `name key=value ...` line on top of `base` / `base_score`.
Variant parseVariantLine(const std::string& line, const TrackerParams& base, float base_score);

/// All variants in `path` (blank lines and '#' comments skipped). An empty
/// path gives the three as-shipped association methods.
std::vector<Variant> loadVariants(const std::string& path, const TrackerParams& base,
                                  float base_score);

}  // namespace s3m::app

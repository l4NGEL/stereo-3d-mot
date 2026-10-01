#include "variants.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace s3m::app {

AssociationMethod parseAssociation(const std::string& s) {
    if (s == "mahalanobis3d")
        return AssociationMethod::kMahalanobis3D;
    if (s == "iou2d")
        return AssociationMethod::kIou2D;
    if (s == "fused")
        return AssociationMethod::kFusedAppearance;
    if (s == "range_gated")
        return AssociationMethod::kRangeGated3D;
    if (s == "two_stage")
        return AssociationMethod::kTwoStage;
    throw std::runtime_error("unknown association '" + s + "'");
}

Variant parseVariantLine(const std::string& line, const TrackerParams& base, float base_score) {
    std::istringstream is(line);
    Variant v;
    v.params = base;
    v.min_score = base_score;
    is >> v.name;
    std::string token;
    while (is >> token) {
        const std::size_t eq = token.find('=');
        if (eq == std::string::npos)
            throw std::runtime_error("variant '" + v.name + "': expected key=value, got '" + token +
                                     "'");
        const std::string key = token.substr(0, eq);
        const std::string val = token.substr(eq + 1);
        if (key == "assoc")
            v.params.association = parseAssociation(val);
        else if (key == "score")
            v.min_score = std::stof(val);
        else if (key == "min_hits")
            v.params.min_hits = std::stoi(val);
        else if (key == "max_age")
            v.params.max_age = std::stoi(val);
        else if (key == "w3d")
            v.params.fused_weight_3d = std::stod(val);
        else if (key == "wiou")
            v.params.fused_weight_iou = std::stod(val);
        else if (key == "wapp")
            v.params.fused_weight_appearance = std::stod(val);
        else if (key == "hungarian")
            v.params.use_hungarian = std::stoi(val) != 0;
        else if (key == "meas_noise")
            v.params.measurement_noise = std::stod(val);
        else if (key == "proc_noise")
            v.params.process_noise = std::stod(val);
        else if (key == "gate_chi2")
            v.params.gating_chi2 = std::stod(val);
        else if (key == "iou_gate")
            v.params.iou_gate = std::stod(val);
        else if (key == "range_gate")
            v.params.range_gate_max_m = std::stod(val);
        else if (key == "iou_pred")
            v.params.iou_use_predicted_box = std::stoi(val) != 0;
        else if (key == "stage1_tsu")
            v.params.stage1_max_tsu = std::stoi(val);
        else if (key == "lost_cost")
            v.params.lost_cost =
                val == "iou"    ? LostTrackCost::kIouPredicted
                : val == "none" ? LostTrackCost::kNone
                : val == "app_mahal"
                    ? LostTrackCost::kAppearanceMahalanobis
                    : throw std::runtime_error("variant '" + v.name + "': unknown lost_cost '" +
                                               val + "'");
        else if (key == "lost_chi2")
            v.params.lost_gating_chi2 = std::stod(val);
        else if (key == "lost_app")
            v.params.lost_max_appearance = std::stod(val);
        else if (key == "reconfirm")
            v.params.reconfirm_after_miss = std::stoi(val) != 0;
        else if (key == "sort_box")
            v.params.sort_box_model = std::stoi(val) != 0;
        else if (key == "max_total")
            v.params.iou_max_total_assignment = std::stoi(val) != 0;
        else if (key == "box_coast")
            v.params.box_coast_frames = std::stoi(val);
        else if (key == "filt_box")
            v.params.report_filtered_box = std::stoi(val) != 0;
        else if (key == "warmup")
            v.params.sort_warmup = std::stoi(val) != 0;
        else
            throw std::runtime_error("variant '" + v.name + "': unknown key '" + key + "'");
    }
    return v;
}

std::vector<Variant> loadVariants(const std::string& path, const TrackerParams& base,
                                  float base_score) {
    std::vector<Variant> out;
    if (path.empty()) {
        const std::pair<const char*, AssociationMethod> shipped[] = {
            {"shipped_mahalanobis3d", AssociationMethod::kMahalanobis3D},
            {"shipped_iou2d", AssociationMethod::kIou2D},
            {"shipped_fused", AssociationMethod::kFusedAppearance},
        };
        for (const auto& s : shipped) {
            Variant v;
            v.name = s.first;
            v.params = base;
            v.params.association = s.second;
            v.min_score = base_score;
            out.push_back(v);
        }
        return out;
    }
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("cannot open variants file '" + path + "'");
    std::string line;
    while (std::getline(f, line)) {
        const std::size_t first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos || line[first] == '#')
            continue;
        out.push_back(parseVariantLine(line, base, base_score));
    }
    return out;
}

}  // namespace s3m::app

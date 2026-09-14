#include "s3m/tracking/tracker.hpp"

#include <algorithm>
#include <limits>
#include <utility>

#include "s3m/detection/nms.hpp"  // reuse the tested iou() for kIou2D cost
#include "s3m/tracking/appearance.hpp"
#include "s3m/tracking/assignment.hpp"

namespace s3m {
namespace {

bool classCompatible(int a, int b) { return a < 0 || b < 0 || a == b; }

}  // namespace

TrackerParams TrackerParams::fromConfig(const TrackingParams& tp) {
    TrackerParams p;
    p.dt = tp.dt;
    p.process_noise = tp.process_noise;
    p.measurement_noise = tp.measurement_noise;
    p.max_age = tp.max_age;
    p.min_hits = tp.min_hits;
    p.association = (tp.association == "iou2d")   ? AssociationMethod::kIou2D
                    : (tp.association == "fused")  ? AssociationMethod::kFusedAppearance
                                                    : AssociationMethod::kMahalanobis3D;
    p.gating_chi2 = tp.gating_chi2;
    p.iou_gate = tp.iou_gate;
    p.use_hungarian = tp.use_hungarian;
    p.fused_weight_3d = tp.fused_weight_3d;
    p.fused_weight_iou = tp.fused_weight_iou;
    p.fused_weight_appearance = tp.fused_weight_appearance;
    return p;
}

Tracker::Tracker(TrackerParams params) : params_(params) {}

void Tracker::reset() {
    tracks_.clear();
    next_id_ = 1;
}

double Tracker::gate() const {
    if (params_.association == AssociationMethod::kMahalanobis3D) return params_.gating_chi2;
    if (params_.association == AssociationMethod::kIou2D) return 1.0 - params_.iou_gate;
    // kFusedAppearance: feasibility is decided per-pair in buildCostMatrix (the
    // union of the two geometric gates), so every cost left in the matrix is
    // already a feasible weighted blend in [0, 1] -- the solver's own gate
    // just needs to not reject those.
    return 1.0;
}

std::vector<std::vector<double>> Tracker::buildCostMatrix(
    const std::vector<int>& usable_dets, const std::vector<Detection3D>& detections) const {
    const std::size_t rows = tracks_.size();
    const std::size_t cols = usable_dets.size();
    std::vector<std::vector<double>> cost(
        rows, std::vector<double>(cols, std::numeric_limits<double>::infinity()));

    for (std::size_t i = 0; i < rows; ++i) {
        const Track& t = tracks_[i];
        for (std::size_t k = 0; k < cols; ++k) {
            const Detection3D& d = detections[static_cast<std::size_t>(usable_dets[k])];
            if (!classCompatible(t.classId(), d.class_id)) continue;

            if (params_.association == AssociationMethod::kMahalanobis3D) {
                cost[i][k] = t.gatingDistanceSq(d.position);
            } else if (params_.association == AssociationMethod::kIou2D) {
                cost[i][k] = 1.0 - iou(t.lastBox(), d.box);
            } else {
                const double mahal_sq = t.gatingDistanceSq(d.position);
                const double iou_val = iou(t.lastBox(), d.box);
                if (mahal_sq > params_.gating_chi2 && iou_val < params_.iou_gate) continue;  // +inf

                const double m_norm = std::min(mahal_sq / params_.gating_chi2, 1.0);
                const double iou_cost = 1.0 - iou_val;
                const double a_cost = (!t.appearance().empty() && !d.appearance.empty())
                                           ? appearanceDistance(t.appearance(), d.appearance)
                                           : 0.5;  // no descriptor on one side -- neutral, not a penalty
                cost[i][k] = params_.fused_weight_3d * m_norm + params_.fused_weight_iou * iou_cost +
                             params_.fused_weight_appearance * a_cost;
            }
        }
    }
    return cost;
}

TrackerUpdateResult Tracker::update(const std::vector<Detection3D>& detections) {
    for (Track& t : tracks_) t.predict();

    std::vector<int> usable;
    usable.reserve(detections.size());
    for (int k = 0; k < static_cast<int>(detections.size()); ++k) {
        if (detections[static_cast<std::size_t>(k)].valid) usable.push_back(k);
    }

    const std::vector<std::vector<double>> cost = buildCostMatrix(usable, detections);
    const Assignment assign = params_.use_hungarian ? solveAssignmentHungarian(cost, gate())
                                                     : solveAssignmentGreedy(cost, gate());

    TrackerUpdateResult result;
    result.detection_track_id.assign(detections.size(), -1);

    std::vector<char> det_matched(usable.size(), 0);
    for (std::size_t i = 0; i < tracks_.size(); ++i) {
        const int k = assign.row_to_col[i];
        if (k < 0) {
            tracks_[i].markMissed();
            continue;
        }
        const int det_idx = usable[static_cast<std::size_t>(k)];
        tracks_[i].correct(detections[static_cast<std::size_t>(det_idx)]);
        det_matched[static_cast<std::size_t>(k)] = 1;
        result.detection_track_id[static_cast<std::size_t>(det_idx)] = tracks_[i].id();
    }

    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                 [this](const Track& t) {
                                     return t.timeSinceUpdate() > params_.max_age;
                                 }),
                 tracks_.end());

    for (std::size_t k = 0; k < usable.size(); ++k) {
        if (det_matched[k]) continue;
        const int det_idx = usable[k];
        const Detection3D& d = detections[static_cast<std::size_t>(det_idx)];
        Track born(next_id_++, d.position, params_.dt, params_.process_noise,
                   params_.measurement_noise, d.class_id, d.box, d.appearance);
        born.setMinHits(params_.min_hits);
        result.detection_track_id[static_cast<std::size_t>(det_idx)] = born.id();
        tracks_.push_back(std::move(born));
    }

    result.tracks.reserve(tracks_.size());
    for (const Track& t : tracks_) result.tracks.push_back(t.snapshot());
    return result;
}

}  // namespace s3m

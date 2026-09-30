#include "s3m/tracking/tracker.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "s3m/detection/nms.hpp"  // reuse the tested iou() for kIou2D cost
#include "s3m/tracking/appearance.hpp"
#include "s3m/tracking/assignment.hpp"

namespace s3m {
namespace {

bool classCompatible(int a, int b) {
    return a < 0 || b < 0 || a == b;
}

}  // namespace

TrackerParams TrackerParams::fromConfig(const TrackingParams& tp) {
    TrackerParams p;
    p.dt = tp.dt;
    p.process_noise = tp.process_noise;
    p.measurement_noise = tp.measurement_noise;
    p.max_age = tp.max_age;
    p.min_hits = tp.min_hits;
    p.association = (tp.association == "iou2d")         ? AssociationMethod::kIou2D
                    : (tp.association == "fused")       ? AssociationMethod::kFusedAppearance
                    : (tp.association == "range_gated") ? AssociationMethod::kRangeGated3D
                    : (tp.association == "two_stage")   ? AssociationMethod::kTwoStage
                                                        : AssociationMethod::kMahalanobis3D;
    p.gating_chi2 = tp.gating_chi2;
    p.iou_gate = tp.iou_gate;
    p.use_hungarian = tp.use_hungarian;
    p.fused_weight_3d = tp.fused_weight_3d;
    p.fused_weight_iou = tp.fused_weight_iou;
    p.fused_weight_appearance = tp.fused_weight_appearance;
    p.range_gate_max_m = tp.range_gate_max_m;
    p.iou_use_predicted_box = tp.iou_use_predicted_box;
    p.stage1_max_tsu = tp.stage1_max_tsu;
    p.lost_cost = (tp.lost_cost == "iou")    ? LostTrackCost::kIouPredicted
                  : (tp.lost_cost == "none") ? LostTrackCost::kNone
                                             : LostTrackCost::kAppearanceMahalanobis;
    p.lost_gating_chi2 = tp.lost_gating_chi2;
    p.lost_max_appearance = tp.lost_max_appearance;
    p.reconfirm_after_miss = tp.reconfirm_after_miss;
    return p;
}

Tracker::Tracker(TrackerParams params) : params_(params) {}

void Tracker::reset() {
    tracks_.clear();
    next_id_ = 1;
}

double Tracker::gate() const {
    if (params_.association == AssociationMethod::kMahalanobis3D)
        return params_.gating_chi2;
    if (params_.association == AssociationMethod::kIou2D ||
        params_.association == AssociationMethod::kTwoStage)
        return 1.0 - params_.iou_gate;
    if (params_.association == AssociationMethod::kRangeGated3D) {
        // Every feasible pair's cost is already normalised to [0, 1] --
        // mahal_sq / gating_chi2 for in-range pairs (clamped, though feasible
        // ones never reach the clamp -- see buildCostMatrix), 1 - iou for
        // out-of-range ones -- so one shared gate works for both branches.
        return 1.0;
    }
    // kFusedAppearance: feasibility is decided per-pair in buildCostMatrix (the
    // union of the two geometric gates), so the solver's own gate just needs
    // to not reject those. The max any single cost can reach is the sum of
    // the three weights (each term is a normalised cost in [0, 1]) -- computed
    // here rather than assumed to be 1.0, since nothing enforces that the
    // weights are normalised.
    return params_.fused_weight_3d + params_.fused_weight_iou + params_.fused_weight_appearance;
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
            if (!classCompatible(t.classId(), d.class_id))
                continue;

            if (params_.association == AssociationMethod::kMahalanobis3D) {
                cost[i][k] = t.gatingDistanceSq(d.position);
            } else if (params_.association == AssociationMethod::kIou2D) {
                const cv::Rect2f box =
                    params_.iou_use_predicted_box ? t.predictedBox() : t.lastBox();
                cost[i][k] = 1.0 - iou(box, d.box);
            } else if (params_.association == AssociationMethod::kRangeGated3D) {
                // Per-pair choice, not a blend: within range, exactly
                // kMahalanobis3D's cost/gate (normalised so it shares a scale
                // with the fallback); beyond it, exactly kIou2D's.
                if (d.position.z <= params_.range_gate_max_m) {
                    const double mahal_sq = t.gatingDistanceSq(d.position);
                    if (mahal_sq > params_.gating_chi2)
                        continue;  // +inf, same feasibility rule as kMahalanobis3D
                    cost[i][k] = std::min(mahal_sq / params_.gating_chi2, 1.0);
                } else {
                    const double iou_val = iou(t.lastBox(), d.box);
                    if (iou_val < params_.iou_gate)
                        continue;  // +inf, same feasibility rule as kIou2D
                    cost[i][k] = 1.0 - iou_val;
                }
            } else {
                const double mahal_sq = t.gatingDistanceSq(d.position);
                const double iou_val = iou(t.lastBox(), d.box);
                if (mahal_sq > params_.gating_chi2 && iou_val < params_.iou_gate)
                    continue;  // +inf

                const double m_norm = std::min(mahal_sq / params_.gating_chi2, 1.0);
                const double iou_cost = 1.0 - iou_val;
                const double a_cost =
                    (!t.appearance().empty() && !d.appearance.empty())
                        ? appearanceDistance(t.appearance(), d.appearance)
                        : 0.5;  // no descriptor on one side -- neutral, not a penalty
                cost[i][k] = params_.fused_weight_3d * m_norm +
                             params_.fused_weight_iou * iou_cost +
                             params_.fused_weight_appearance * a_cost;
            }
        }
    }
    return cost;
}

double Tracker::lostPairCost(const Track& t, const Detection3D& d) const {
    constexpr double kInf = std::numeric_limits<double>::infinity();
    switch (params_.lost_cost) {
        case LostTrackCost::kNone:
            return kInf;
        case LostTrackCost::kIouPredicted: {
            const double iou_val = iou(t.predictedBox(), d.box);
            return iou_val < params_.iou_gate ? kInf : 1.0 - iou_val;
        }
        case LostTrackCost::kAppearanceMahalanobis: {
            if (t.appearance().empty() || d.appearance.empty())
                return kInf;  // no descriptor -> not recoverable by this rule
            const double mahal_sq = t.gatingDistanceSq(d.position);
            if (!std::isfinite(mahal_sq) || mahal_sq < 0.0 || mahal_sq > params_.lost_gating_chi2)
                return kInf;
            const double app = appearanceDistance(t.appearance(), d.appearance);
            if (!std::isfinite(app) || app > params_.lost_max_appearance)
                return kInf;
            return 0.5 * mahal_sq / params_.lost_gating_chi2 +
                   0.5 * app / params_.lost_max_appearance;
        }
    }
    return kInf;
}

void Tracker::associateTwoStage(const std::vector<int>& usable,
                                const std::vector<Detection3D>& detections,
                                std::vector<int>& track_to_det,
                                std::vector<int>& stage_of_track) const {
    constexpr double kInf = std::numeric_limits<double>::infinity();
    track_to_det.assign(tracks_.size(), -1);
    stage_of_track.assign(tracks_.size(), 0);
    const auto solve = [this](const std::vector<std::vector<double>>& c, double g) {
        return params_.use_hungarian ? solveAssignmentHungarian(c, g) : solveAssignmentGreedy(c, g);
    };

    // time_since_update is only advanced by markMissed() after association, so
    // here it still counts the frames missed *before* this one -- predict()
    // (which already ran) advances age, never time_since_update.
    std::vector<int> active;
    std::vector<int> lost;
    for (int i = 0; i < static_cast<int>(tracks_.size()); ++i) {
        if (tracks_[static_cast<std::size_t>(i)].timeSinceUpdate() <= params_.stage1_max_tsu)
            active.push_back(i);
        else
            lost.push_back(i);
    }

    // Stage 1: active tracks x every usable detection, predicted-box IoU.
    std::vector<char> det_taken(usable.size(), 0);
    if (!active.empty() && !usable.empty()) {
        std::vector<std::vector<double>> cost(active.size(),
                                              std::vector<double>(usable.size(), kInf));
        for (std::size_t r = 0; r < active.size(); ++r) {
            const Track& t = tracks_[static_cast<std::size_t>(active[r])];
            const cv::Rect2f pbox = t.predictedBox();
            for (std::size_t k = 0; k < usable.size(); ++k) {
                const Detection3D& d = detections[static_cast<std::size_t>(usable[k])];
                if (!classCompatible(t.classId(), d.class_id))
                    continue;
                const double iou_val = iou(pbox, d.box);
                if (iou_val >= params_.iou_gate)
                    cost[r][k] = 1.0 - iou_val;
            }
        }
        const Assignment a = solve(cost, 1.0 - params_.iou_gate);
        for (std::size_t r = 0; r < active.size(); ++r) {
            const int k = a.row_to_col[r];
            if (k < 0)
                continue;
            track_to_det[static_cast<std::size_t>(active[r])] = k;
            stage_of_track[static_cast<std::size_t>(active[r])] = 1;
            det_taken[static_cast<std::size_t>(k)] = 1;
        }
    }

    // Stage 2: lost tracks x the detections stage 1 did not take. Stage-1
    // matches are final: those detections are simply not columns here.
    std::vector<int> left;  // indices into `usable`
    for (std::size_t k = 0; k < usable.size(); ++k) {
        if (!det_taken[k])
            left.push_back(static_cast<int>(k));
    }
    if (lost.empty() || left.empty() || params_.lost_cost == LostTrackCost::kNone)
        return;
    std::vector<std::vector<double>> cost(lost.size(), std::vector<double>(left.size(), kInf));
    for (std::size_t r = 0; r < lost.size(); ++r) {
        const Track& t = tracks_[static_cast<std::size_t>(lost[r])];
        for (std::size_t c = 0; c < left.size(); ++c) {
            const Detection3D& d =
                detections[static_cast<std::size_t>(usable[static_cast<std::size_t>(left[c])])];
            if (classCompatible(t.classId(), d.class_id))
                cost[r][c] = lostPairCost(t, d);
        }
    }
    // Every feasible stage-2 cost lies in [0, 1]: IoU gives 1 - iou <= 1 - iou_gate,
    // appearance+Mahalanobis the mean of two gate-normalised terms, each <= 1.
    const Assignment a = solve(cost, 1.0);
    for (std::size_t r = 0; r < lost.size(); ++r) {
        const int c = a.row_to_col[r];
        if (c < 0)
            continue;
        track_to_det[static_cast<std::size_t>(lost[r])] = left[static_cast<std::size_t>(c)];
        stage_of_track[static_cast<std::size_t>(lost[r])] = 2;
    }
}

TrackerUpdateResult Tracker::update(const std::vector<Detection3D>& detections) {
    for (Track& t : tracks_)
        t.predict();

    std::vector<int> usable;
    usable.reserve(detections.size());
    for (int k = 0; k < static_cast<int>(detections.size()); ++k) {
        if (detections[static_cast<std::size_t>(k)].valid)
            usable.push_back(k);
    }

    // track_to_det: per tracks_ index, an index into `usable`, or -1.
    std::vector<int> track_to_det;
    std::vector<int> stage_of_track;
    if (params_.association == AssociationMethod::kTwoStage) {
        associateTwoStage(usable, detections, track_to_det, stage_of_track);
    } else {
        const std::vector<std::vector<double>> cost = buildCostMatrix(usable, detections);
        const Assignment assign = params_.use_hungarian ? solveAssignmentHungarian(cost, gate())
                                                        : solveAssignmentGreedy(cost, gate());
        track_to_det = assign.row_to_col;
        stage_of_track.assign(tracks_.size(), 0);
        for (std::size_t i = 0; i < tracks_.size(); ++i) {
            if (track_to_det[i] >= 0)
                stage_of_track[i] = 1;
        }
    }

    TrackerUpdateResult result;
    result.detection_track_id.assign(detections.size(), -1);

    if (params_.collect_diagnostics) {
        for (std::size_t i = 0; i < tracks_.size(); ++i) {
            const Track& t = tracks_[i];
            const cv::Rect2f pbox = t.predictedBox();
            for (std::size_t k = 0; k < usable.size(); ++k) {
                const Detection3D& d = detections[static_cast<std::size_t>(usable[k])];
                if (!classCompatible(t.classId(), d.class_id))
                    continue;
                const bool matched = track_to_det[i] == static_cast<int>(k);
                AssociationDiagnostic g;
                g.iou_last = iou(t.lastBox(), d.box);
                g.iou_predicted = iou(pbox, d.box);
                if (!matched && g.iou_last <= 0.0 && g.iou_predicted <= 0.0)
                    continue;
                g.track_id = t.id();
                g.detection_index = usable[k];
                g.tsu_before = t.timeSinceUpdate();
                g.confirmed_before = t.confirmed();
                g.mahalanobis_sq = t.gatingDistanceSq(d.position);
                g.appearance = (!t.appearance().empty() && !d.appearance.empty())
                                   ? appearanceDistance(t.appearance(), d.appearance)
                                   : -1.0;
                g.matched_stage = matched ? stage_of_track[i] : 0;
                result.diagnostics.push_back(g);
            }
        }
    }

    std::vector<char> det_matched(usable.size(), 0);
    for (std::size_t i = 0; i < tracks_.size(); ++i) {
        const int k = track_to_det[i];
        if (k < 0) {
            tracks_[i].markMissed();
            continue;
        }
        const int det_idx = usable[static_cast<std::size_t>(k)];
        tracks_[i].correct(detections[static_cast<std::size_t>(det_idx)]);
        det_matched[static_cast<std::size_t>(k)] = 1;
        result.detection_track_id[static_cast<std::size_t>(det_idx)] = tracks_[i].id();
    }

    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
                       [this](const Track& t) { return t.timeSinceUpdate() > params_.max_age; }),
        tracks_.end());

    for (std::size_t k = 0; k < usable.size(); ++k) {
        if (det_matched[k])
            continue;
        const int det_idx = usable[k];
        const Detection3D& d = detections[static_cast<std::size_t>(det_idx)];
        Track born(next_id_++, d.position, params_.dt, params_.process_noise,
                   params_.measurement_noise, d.class_id, d.box, d.appearance);
        born.setMinHits(params_.min_hits);
        born.setReconfirmAfterMiss(params_.reconfirm_after_miss);
        result.detection_track_id[static_cast<std::size_t>(det_idx)] = born.id();
        tracks_.push_back(std::move(born));
    }

    result.tracks.reserve(tracks_.size());
    for (const Track& t : tracks_)
        result.tracks.push_back(t.snapshot());
    return result;
}

}  // namespace s3m

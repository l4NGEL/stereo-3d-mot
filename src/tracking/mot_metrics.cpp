#include "s3m/tracking/mot_metrics.hpp"

#include <cmath>
#include <limits>
#include <sstream>
#include <utility>

#include "s3m/tracking/assignment.hpp"

namespace s3m {
namespace {

double distance3(const cv::Point3f& a, const cv::Point3f& b) {
    const double dx = static_cast<double>(a.x) - b.x;
    const double dy = static_cast<double>(a.y) - b.y;
    const double dz = static_cast<double>(a.z) - b.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

}  // namespace

MotAccumulator::MotAccumulator(double max_dist) : max_dist_(max_dist) {}

void MotAccumulator::reset() {
    frame_index_ = 0;
    prev_match_.clear();
    events_by_gt_.clear();
    co_occurrence_.clear();
    gt_frame_count_.clear();
    hyp_frame_count_.clear();
    num_matches_ = num_switches_ = num_misses_ = num_fp_ = num_objects_ = 0;
    motp_sum_ = 0.0;
}

void MotAccumulator::update(const std::vector<MotObject>& gt, const std::vector<MotObject>& hyp) {
    const std::size_t n = gt.size();
    const std::size_t m = hyp.size();
    num_objects_ += static_cast<std::int64_t>(n);

    constexpr double kInf = std::numeric_limits<double>::infinity();
    std::vector<std::vector<double>> dist(n, std::vector<double>(m, kInf));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
            const double d = distance3(gt[i].position, hyp[j].position);
            if (d <= max_dist_) {
                dist[i][j] = d;
                ++co_occurrence_[std::make_pair(gt[i].id, hyp[j].id)];
            }
        }
    }
    for (const MotObject& g : gt)
        ++gt_frame_count_[g.id];
    for (const MotObject& h : hyp)
        ++hyp_frame_count_[h.id];

    std::vector<char> gt_used(n, 0);
    std::vector<char> hyp_used(m, 0);
    std::map<int, std::size_t> hyp_index;
    for (std::size_t j = 0; j < m; ++j)
        hyp_index[hyp[j].id] = j;

    // Step 1: re-establish each GT id's persistent previous correspondence.
    for (std::size_t i = 0; i < n; ++i) {
        const auto pm = prev_match_.find(gt[i].id);
        if (pm == prev_match_.end())
            continue;
        const auto hj = hyp_index.find(pm->second);
        if (hj == hyp_index.end())
            continue;
        const std::size_t j = hj->second;
        if (hyp_used[j] || dist[i][j] == kInf)
            continue;

        gt_used[i] = 1;
        hyp_used[j] = 1;
        ++num_matches_;
        motp_sum_ += dist[i][j];
        events_by_gt_[gt[i].id].push_back(ObjEvent{frame_index_, true});
    }

    // Step 2: Hungarian-match the remainder, gated at max_dist_.
    std::vector<std::size_t> rem_i;
    std::vector<std::size_t> rem_j;
    for (std::size_t i = 0; i < n; ++i) {
        if (!gt_used[i])
            rem_i.push_back(i);
    }
    for (std::size_t j = 0; j < m; ++j) {
        if (!hyp_used[j])
            rem_j.push_back(j);
    }
    if (!rem_i.empty() && !rem_j.empty()) {
        std::vector<std::vector<double>> sub(rem_i.size(), std::vector<double>(rem_j.size()));
        for (std::size_t ri = 0; ri < rem_i.size(); ++ri) {
            for (std::size_t rj = 0; rj < rem_j.size(); ++rj) {
                sub[ri][rj] = dist[rem_i[ri]][rem_j[rj]];
            }
        }
        const Assignment assign = solveAssignmentHungarian(sub, max_dist_);
        for (std::size_t ri = 0; ri < rem_i.size(); ++ri) {
            const int rj = assign.row_to_col[ri];
            if (rj < 0)
                continue;
            const std::size_t i = rem_i[ri];
            const std::size_t j = rem_j[static_cast<std::size_t>(rj)];
            const int g = gt[i].id;
            const int h = hyp[j].id;

            const auto pm = prev_match_.find(g);
            const bool is_switch = (pm != prev_match_.end()) && (pm->second != h);
            if (is_switch) {
                ++num_switches_;
            } else {
                ++num_matches_;
            }
            events_by_gt_[g].push_back(ObjEvent{frame_index_, true});
            motp_sum_ += sub[ri][static_cast<std::size_t>(rj)];
            prev_match_[g] = h;
            gt_used[i] = 1;
            hyp_used[j] = 1;
        }
    }

    // Steps 3-4: whatever is left is a miss (FN) or a false positive (FP).
    for (std::size_t i = 0; i < n; ++i) {
        if (gt_used[i])
            continue;
        ++num_misses_;
        events_by_gt_[gt[i].id].push_back(ObjEvent{frame_index_, false});
    }
    for (std::size_t j = 0; j < m; ++j) {
        if (!hyp_used[j])
            ++num_fp_;
    }

    ++frame_index_;
}

MotSummary MotAccumulator::summary() const {
    MotSummary s;
    s.num_matches = num_matches_;
    s.num_switches = num_switches_;
    s.num_misses = num_misses_;
    s.num_false_positives = num_fp_;
    s.num_objects = num_objects_;

    const std::int64_t num_detections = num_matches_ + num_switches_;
    s.mota = num_objects_ > 0 ? 1.0 - static_cast<double>(num_misses_ + num_switches_ + num_fp_) /
                                          static_cast<double>(num_objects_)
                              : 0.0;
    s.motp = num_detections > 0 ? motp_sum_ / static_cast<double>(num_detections) : 0.0;
    s.precision = (num_fp_ + num_detections) > 0 ? static_cast<double>(num_detections) /
                                                       static_cast<double>(num_fp_ + num_detections)
                                                 : 0.0;
    s.recall = num_objects_ > 0
                   ? static_cast<double>(num_detections) / static_cast<double>(num_objects_)
                   : 0.0;

    // Fragmentation: per GT id, restricted to [first match, last match] --
    // trailing/leading misses outside that span don't count (see the
    // gate-before-padding-style rationale in mot_metrics.hpp's docstring: this
    // mirrors motmetrics' num_fragmentations() exactly).
    std::int64_t frag = 0;
    for (const auto& gt_events : events_by_gt_) {
        const std::vector<ObjEvent>& evs = gt_events.second;
        int first = -1;
        int last = -1;
        for (std::size_t k = 0; k < evs.size(); ++k) {
            if (evs[k].is_match) {
                if (first < 0)
                    first = static_cast<int>(k);
                last = static_cast<int>(k);
            }
        }
        if (first < 0)
            continue;
        bool have_prev = false;
        bool prev_is_miss = false;
        for (int k = first; k <= last; ++k) {
            const bool is_miss = !evs[static_cast<std::size_t>(k)].is_match;
            if (have_prev && !prev_is_miss && is_miss)
                ++frag;
            prev_is_miss = is_miss;
            have_prev = true;
        }
    }
    s.num_fragmentations = frag;

    // IDF1: a *global* trajectory-level assignment maximising total gated
    // co-occurrence. Every (gt, hyp) pair is a candidate (gate = 0.0 admits
    // exactly the non-positive costs below -- i.e. every pair); extra
    // zero-benefit pairings the solver is forced to include past the truly
    // useful ones don't change the resulting idtp.
    std::vector<int> gt_ids;
    std::vector<int> hyp_ids;
    for (const auto& id_cnt : gt_frame_count_)
        gt_ids.push_back(id_cnt.first);
    for (const auto& id_cnt : hyp_frame_count_)
        hyp_ids.push_back(id_cnt.first);

    std::int64_t idtp = 0;
    if (!gt_ids.empty() && !hyp_ids.empty()) {
        std::vector<std::vector<double>> cost(gt_ids.size(),
                                              std::vector<double>(hyp_ids.size(), 0.0));
        for (std::size_t i = 0; i < gt_ids.size(); ++i) {
            for (std::size_t j = 0; j < hyp_ids.size(); ++j) {
                const auto it = co_occurrence_.find(std::make_pair(gt_ids[i], hyp_ids[j]));
                cost[i][j] = it != co_occurrence_.end() ? -static_cast<double>(it->second) : 0.0;
            }
        }
        const Assignment assign = solveAssignmentHungarian(cost, 0.0);
        for (std::size_t i = 0; i < assign.row_to_col.size(); ++i) {
            const int j = assign.row_to_col[i];
            if (j < 0)
                continue;
            const auto it = co_occurrence_.find(
                std::make_pair(gt_ids[i], hyp_ids[static_cast<std::size_t>(j)]));
            if (it != co_occurrence_.end())
                idtp += it->second;
        }
    }
    s.idtp = idtp;

    std::int64_t total_gt_frames = 0;
    std::int64_t total_hyp_frames = 0;
    for (const auto& id_cnt : gt_frame_count_)
        total_gt_frames += id_cnt.second;
    for (const auto& id_cnt : hyp_frame_count_)
        total_hyp_frames += id_cnt.second;
    s.idfn = total_gt_frames - idtp;
    s.idfp = total_hyp_frames - idtp;
    const std::int64_t idf1_denom = 2 * idtp + s.idfp + s.idfn;
    s.idf1 =
        idf1_denom > 0 ? 2.0 * static_cast<double>(idtp) / static_cast<double>(idf1_denom) : 0.0;

    return s;
}

std::string MotSummary::toString() const {
    std::ostringstream os;
    os << "MOTA=" << mota << " MOTP=" << motp << " IDF1=" << idf1 << " IDSW=" << num_switches
       << " Frag=" << num_fragmentations << " FP=" << num_false_positives << " FN=" << num_misses
       << " Prec=" << precision << " Rec=" << recall << " (objects=" << num_objects << ")";
    return os.str();
}

}  // namespace s3m

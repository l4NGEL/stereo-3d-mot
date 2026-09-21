#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace s3m {

/// One ground-truth or hypothesis (predicted) object at a single frame, for
/// MOT evaluation. `position` is compared with plain Euclidean distance in
/// whatever space the caller puts it in -- 3D camera-frame metres for the
/// project's own Tracker output, or e.g. (x, y, 0) image pixels for a 2D
/// IoU-free comparison.
struct MotObject {
    int id = -1;
    cv::Point3f position{};
};

/// Aggregated CLEAR-MOT (Bernardin & Stiefelhagen, 2008) + IDF1
/// (Ristani et al., 2016) metrics.
struct MotSummary {
    double mota = 0.0;  ///< 1 - (misses + switches + false_positives) / objects
    double motp = 0.0;  ///< mean distance over matched pairs (lower is better)
    double idf1 = 0.0;  ///< identity F1: 2*idtp / (2*idtp + idfp + idfn)
    std::int64_t idtp = 0;
    std::int64_t idfp = 0;
    std::int64_t idfn = 0;
    std::int64_t num_matches = 0;         ///< matched, same id as last time it was seen
    std::int64_t num_switches = 0;        ///< matched, but a DIFFERENT id than last time
    std::int64_t num_fragmentations = 0;  ///< times a GT track went tracked -> not tracked
    std::int64_t num_misses = 0;          ///< false negatives
    std::int64_t num_false_positives = 0;
    std::int64_t num_objects = 0;  ///< total GT appearances across all frames
    double precision = 0.0;
    double recall = 0.0;

    /// One-line "mota=.. motp=.. idf1=.. ..." summary.
    std::string toString() const;
};

/// CLEAR-MOT + IDF1 accumulator. Feed it one frame at a time via update(), in
/// increasing frame order, then read summary().
///
/// Per-frame matching: (1) try to re-establish each GT id's *previous*
/// hypothesis id, if that same hypothesis id is present again this frame and
/// within `max_dist` -- this persists across any number of frames where the
/// GT id was simply absent, not just a one-frame gap; (2) Hungarian-match
/// everything left over (via solveAssignmentHungarian), gated at `max_dist`.
/// A step-2 pairing that contradicts a GT id's previous hypothesis id is an
/// ID SWITCH rather than a plain match. This mirrors py-motmetrics'
/// `MOTAccumulator` exactly -- validated against it directly (400+ randomised
/// trials plus hand-built edge cases: ties, gaps, crossings, gating) before
/// this port; see tests/test_mot_metrics.cpp.
class MotAccumulator {
 public:
    explicit MotAccumulator(double max_dist);

    void update(const std::vector<MotObject>& gt, const std::vector<MotObject>& hyp);

    MotSummary summary() const;
    void reset();

 private:
    struct ObjEvent {
        int frame;
        bool is_match;  ///< false -> this frame was a MISS for this GT id
    };

    double max_dist_;
    int frame_index_ = 0;

    std::map<int, int> prev_match_;                              ///< gt id -> hyp id, persistent
    std::map<int, std::vector<ObjEvent>> events_by_gt_;          ///< for fragmentation
    std::map<std::pair<int, int>, std::int64_t> co_occurrence_;  ///< (gt,hyp) -> gated co-frames
    std::map<int, std::int64_t> gt_frame_count_;
    std::map<int, std::int64_t> hyp_frame_count_;

    std::int64_t num_matches_ = 0;
    std::int64_t num_switches_ = 0;
    std::int64_t num_misses_ = 0;
    std::int64_t num_fp_ = 0;
    std::int64_t num_objects_ = 0;
    double motp_sum_ = 0.0;
};

}  // namespace s3m

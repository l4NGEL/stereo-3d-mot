#pragma once

#include <vector>

#include "s3m/core/config.hpp"
#include "s3m/core/types.hpp"
#include "s3m/tracking/track.hpp"

namespace s3m {

/// How predicted tracks are matched against new detections. Both methods only
/// ever consider detections with a valid depth-derived 3D position (`valid`
/// from promoteTo3D), so every Track's Kalman filter is always corrected with a
/// real 3D measurement; the difference is purely which cue the *association
/// decision* uses.
enum class AssociationMethod {
    /// Squared Mahalanobis distance in the track's 3D state space, gated by a
    /// chi-square threshold. Two objects that overlap on screen but sit at
    /// different depths are not confused -- the metric distance separates them.
    kMahalanobis3D,
    /// 1 - IoU between the track's last matched 2D box and the detection's box.
    /// The classic baseline: ignores depth for the decision entirely, so two
    /// objects at different depths that happen to overlap in the image can be
    /// swapped. Kept to make that failure mode demonstrable (see
    /// benchmark_track and TrackerTest.DepthSeparatesOccludingBoxes).
    kIou2D,
    /// Weighted fusion of all three cues: normalised squared Mahalanobis
    /// distance, 1 - IoU, and appearance (Bhattacharyya on an HSV histogram,
    /// tracking/appearance.hpp) -- `TrackerParams::fused_weight_*`. Gated on
    /// the *union* of the Mahalanobis and IoU gates (either geometric cue
    /// admitting the pair is enough), not their intersection: the point of
    /// fusing in a second, independent cue is to stay robust when one of them
    /// is unreliable, and gating on the intersection would just inherit
    /// whichever cue is currently worse. A detection with no appearance
    /// descriptor (or a track that hasn't matched one yet) falls back to a
    /// neutral 0.5 appearance cost rather than being excluded.
    kFusedAppearance,
};

struct TrackerParams {
    double dt = 0.1;
    double process_noise = 1.0;
    double measurement_noise = 0.05;
    int max_age = 30;
    int min_hits = 3;
    AssociationMethod association = AssociationMethod::kMahalanobis3D;
    double gating_chi2 = 7.815;  ///< kMahalanobis3D: chi-square(3 dof, 95%)
    double iou_gate = 0.3;       ///< kIou2D: minimum IoU to allow a match
    bool use_hungarian = true;   ///< false -> greedy nearest-first

    /// kFusedAppearance cost weights; should sum to 1 (not enforced -- an
    /// unnormalised sum just rescales the gate=1.0 comparison uniformly,
    /// which is harmless, but the individual terms stop being readable as
    /// "share of the cost").
    double fused_weight_3d = 0.5;
    double fused_weight_iou = 0.2;
    double fused_weight_appearance = 0.3;

    static TrackerParams fromConfig(const TrackingParams& tp);
};

/// Result of one Tracker::update() call.
struct TrackerUpdateResult {
    std::vector<TrackState> tracks;  ///< every surviving track, confirmed and tentative
    /// Parallel to the `detections` passed to update(): the id of the track
    /// each detection ended up part of (matched to an existing one, or the new
    /// track it birthed), or -1 for a detection that was unusable
    /// (`!Detection3D::valid`) and therefore never touched the tracker.
    std::vector<int> detection_track_id;
};

/// Multi-object tracker: predict every track, associate against this frame's
/// detections, update matches, coast or kill misses, birth new tracks for
/// leftover detections.
///
/// A detection is only ever matched (or used to birth a track) when
/// `Detection3D::valid` is true and its class id is compatible with the
/// track's (class ids differ and neither is < 0 -> never matched, regardless of
/// geometry). Everything else about a frame's detections is ignored.
class Tracker {
 public:
    explicit Tracker(TrackerParams params = TrackerParams());

    /// Advance one frame: predict, associate, update matches, coast/kill
    /// misses, birth new tracks for leftover usable detections.
    TrackerUpdateResult update(const std::vector<Detection3D>& detections);

    const std::vector<Track>& tracks() const { return tracks_; }
    void reset();

 private:
    double gate() const;
    std::vector<std::vector<double>> buildCostMatrix(
        const std::vector<int>& usable_dets, const std::vector<Detection3D>& detections) const;

    TrackerParams params_;
    std::vector<Track> tracks_;
    int next_id_ = 1;
};

}  // namespace s3m

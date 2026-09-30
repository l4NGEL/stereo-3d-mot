#pragma once

#include <opencv2/core.hpp>

#include "s3m/core/types.hpp"
#include "s3m/tracking/kalman_filter.hpp"

namespace s3m {

/// One tracked target: a 3D constant-velocity Kalman filter plus the bookkeeping
/// a tracker-manager needs (age, hit count, misses, confirmation).
class Track {
 public:
    Track(int id, const cv::Point3f& initial_position, double dt, double accel_std, double meas_std,
          int class_id = -1, const cv::Rect2f& initial_box = cv::Rect2f(),
          const cv::Mat& initial_appearance = cv::Mat());

    void predict();

    /// Correct with a bare 3D position (class id / box left unchanged).
    void correct(const cv::Point3f& measured_position);
    /// Correct with a full detection: updates the filter from `position` and
    /// records `box` / `class_id` for display and 2D-IoU association.
    void correct(const Detection3D& detection);

    void markMissed();

    int id() const { return id_; }
    cv::Point3f position() const;
    cv::Point3f velocity() const;
    const cv::Rect2f& lastBox() const { return last_box_; }
    /// This frame's image-plane box prediction (the 2D box filter advanced by
    /// predict(), see makeConstantVelocityBox2D). Kept separate from lastBox(),
    /// which stays the last *observed* box and is what snapshot() reports.
    /// Falls back to lastBox() when the track has no usable box model (born from
    /// an empty box) or the prediction is non-finite; width/height are never
    /// below 1 px.
    cv::Rect2f predictedBox() const;
    int classId() const { return class_id_; }
    /// Running appearance descriptor (EMA over matched detections' -- see
    /// tracking/appearance.hpp), or empty if the track has never matched a
    /// detection that carried one.
    const cv::Mat& appearance() const { return appearance_; }

    int age() const { return age_; }
    int hits() const { return hits_; }
    int timeSinceUpdate() const { return time_since_update_; }
    bool confirmed() const { return confirmed_; }
    void setMinHits(int min_hits) { min_hits_ = min_hits; }
    /// SORT's reporting rule: a miss revokes confirmation, which then needs
    /// `min_hits` *consecutive* matches again. Off (default) = confirmed once,
    /// confirmed for life, the evaluated baseline. Identity, the filters and
    /// the ability to be re-matched are unaffected either way.
    void setReconfirmAfterMiss(bool on) { reconfirm_after_miss_ = on; }
    int hitStreak() const { return hit_streak_; }

    /// Squared Mahalanobis distance from the current prediction to a candidate
    /// measured position (gating for data association).
    double gatingDistanceSq(const cv::Point3f& measured_position) const;

    TrackState snapshot() const;

 private:
    int id_;
    KalmanFilter kf_;
    KalmanFilter box_kf_;
    bool has_box_model_ = false;
    cv::Rect2f last_box_{};
    int class_id_;
    cv::Mat appearance_;
    int age_ = 0;
    int hits_ = 1;
    int time_since_update_ = 0;
    int min_hits_ = 3;
    int hit_streak_ = 1;
    bool reconfirm_after_miss_ = false;
    bool confirmed_ = false;
};

}  // namespace s3m

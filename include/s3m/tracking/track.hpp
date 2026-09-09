#pragma once

#include <opencv2/core.hpp>

#include "s3m/core/types.hpp"
#include "s3m/tracking/kalman_filter.hpp"

namespace s3m {

/// One tracked target: a 3D constant-velocity Kalman filter plus the bookkeeping
/// a tracker-manager needs (age, hit count, misses, confirmation).
class Track {
 public:
    Track(int id, const cv::Point3f& initial_position, double dt, double accel_std,
          double meas_std);

    void predict();
    void correct(const cv::Point3f& measured_position);
    void markMissed();

    int id() const { return id_; }
    cv::Point3f position() const;
    cv::Point3f velocity() const;

    int age() const { return age_; }
    int hits() const { return hits_; }
    int timeSinceUpdate() const { return time_since_update_; }
    bool confirmed() const { return confirmed_; }
    void setMinHits(int min_hits) { min_hits_ = min_hits; }

    /// Squared Mahalanobis distance from the current prediction to a candidate
    /// measured position (gating for data association).
    double gatingDistanceSq(const cv::Point3f& measured_position) const;

    TrackState snapshot() const;

 private:
    int id_;
    KalmanFilter kf_;
    int age_ = 0;
    int hits_ = 1;
    int time_since_update_ = 0;
    int min_hits_ = 3;
    bool confirmed_ = false;
};

}  // namespace s3m

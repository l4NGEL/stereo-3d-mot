#include "s3m/tracking/track.hpp"

namespace s3m {
namespace {

KalmanFilter::Vec toVec3(const cv::Point3f& p) {
    KalmanFilter::Vec v(3);
    v << static_cast<double>(p.x), static_cast<double>(p.y), static_cast<double>(p.z);
    return v;
}

/// Weight kept on a track's existing appearance descriptor each time it's
/// blended with a newly-matched detection's; the new descriptor contributes
/// (1 - kAppearanceEma). Slow-adapting on purpose (DeepSORT-style gallery
/// smoothing) so one noisy or partially-occluded frame can't overwrite a
/// track's identity in a single update.
constexpr double kAppearanceEma = 0.9;

}  // namespace

Track::Track(int id, const cv::Point3f& initial_position, double dt, double accel_std,
             double meas_std, int class_id, const cv::Rect2f& initial_box,
             const cv::Mat& initial_appearance)
    : id_(id), kf_(makeConstantVelocity3D(dt, accel_std, meas_std)), last_box_(initial_box),
      class_id_(class_id),
      // Deep-copy: cv::Mat's copy ctor is a shallow, refcounted alias. Without
      // .clone() here, appearance_ shares the caller's buffer (e.g. a
      // FrameData cached across many Tracker runs in benchmark_kitti
      // --sweep), and correct()'s EMA blend below -- which reassigns
      // appearance_ in place -- would silently corrupt that caller-owned
      // data out from under it, making later, supposedly-independent runs
      // over the same cached frames see already-mutated input. Caught via a
      // determinism check: identical (association, weights) runs gave
      // different results depending on how many other variants had already
      // run first in the same process.
      appearance_(initial_appearance.empty() ? cv::Mat() : initial_appearance.clone()) {
    KalmanFilter::Vec x0 = KalmanFilter::Vec::Zero(6);
    x0(0) = static_cast<double>(initial_position.x);
    x0(1) = static_cast<double>(initial_position.y);
    x0(2) = static_cast<double>(initial_position.z);
    kf_.init(x0, kf_.covariance());
}

void Track::predict() {
    kf_.predict();
    ++age_;
}

void Track::correct(const cv::Point3f& measured_position) {
    kf_.update(toVec3(measured_position));
    ++hits_;
    time_since_update_ = 0;
    if (!confirmed_ && hits_ >= min_hits_) confirmed_ = true;
}

void Track::correct(const Detection3D& detection) {
    correct(detection.position);
    last_box_ = detection.box;
    class_id_ = detection.class_id;
    if (!detection.appearance.empty()) {
        if (appearance_.empty()) {
            appearance_ = detection.appearance.clone();
        } else {
            appearance_ = kAppearanceEma * appearance_ + (1.0 - kAppearanceEma) * detection.appearance;
        }
    }
}

void Track::markMissed() { ++time_since_update_; }

cv::Point3f Track::position() const {
    const KalmanFilter::Vec& x = kf_.state();
    return {static_cast<float>(x(0)), static_cast<float>(x(1)), static_cast<float>(x(2))};
}

cv::Point3f Track::velocity() const {
    const KalmanFilter::Vec& x = kf_.state();
    return {static_cast<float>(x(3)), static_cast<float>(x(4)), static_cast<float>(x(5))};
}

double Track::gatingDistanceSq(const cv::Point3f& measured_position) const {
    return kf_.gatingDistanceSq(toVec3(measured_position));
}

TrackState Track::snapshot() const {
    TrackState state;
    state.id = id_;
    state.position = position();
    state.velocity = velocity();
    state.box = last_box_;
    state.class_id = class_id_;
    state.age = age_;
    state.hits = hits_;
    state.time_since_update = time_since_update_;
    state.confirmed = confirmed_;
    return state;
}

}  // namespace s3m

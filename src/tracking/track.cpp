#include "s3m/tracking/track.hpp"

#include <algorithm>

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
    : id_(id),
      kf_(makeConstantVelocity3D(dt, accel_std, meas_std)),
      last_box_(initial_box),
      class_id_(class_id),
      // Deep-copy: cv::Mat's copy ctor is a shallow, refcounted alias. Without
      // .clone(), appearance_ would share the caller's buffer, and correct()'s
      // in-place EMA blend below would silently mutate caller-owned data
      // (e.g. a detection cached across multiple Tracker runs).
      appearance_(initial_appearance.empty() ? cv::Mat() : initial_appearance.clone()) {
    KalmanFilter::Vec x0 = KalmanFilter::Vec::Zero(6);
    x0(0) = static_cast<double>(initial_position.x);
    x0(1) = static_cast<double>(initial_position.y);
    x0(2) = static_cast<double>(initial_position.z);
    kf_.init(x0, kf_.covariance());
    if (initial_box.width > 0.0f && initial_box.height > 0.0f) {
        box_kf_ = makeConstantVelocityBox2D(initial_box);
        has_box_model_ = true;
    }
}

void Track::predict() {
    kf_.predict();
    if (has_box_model_) {
        // SORT's guard, per dimension: a shrinking size that would cross zero
        // stops shrinking instead of turning the box inside out.
        KalmanFilter::Vec& x = box_kf_.mutableState();
        for (int i = 2; i < 4; ++i) {
            if (x(i) + x(i + 4) <= 1.0)
                x(i + 4) = 0.0;
        }
        box_kf_.predict();
    }
    ++age_;
}

cv::Rect2f Track::predictedBox() const {
    if (!has_box_model_)
        return last_box_;
    const KalmanFilter::Vec& x = box_kf_.state();
    if (!x.head(4).allFinite())
        return last_box_;
    const double w = std::max(x(2), 1.0);
    const double h = std::max(x(3), 1.0);
    return {static_cast<float>(x(0) - w / 2.0), static_cast<float>(x(1) - h / 2.0),
            static_cast<float>(w), static_cast<float>(h)};
}

void Track::correct(const cv::Point3f& measured_position) {
    kf_.update(toVec3(measured_position));
    ++hits_;
    ++hit_streak_;
    time_since_update_ = 0;
    if (reconfirm_after_miss_)
        confirmed_ = hit_streak_ >= min_hits_;
    else if (!confirmed_ && hits_ >= min_hits_)
        confirmed_ = true;
}

void Track::correct(const Detection3D& detection) {
    correct(detection.position);
    if (detection.box.width > 0.0f && detection.box.height > 0.0f) {
        KalmanFilter::Vec z(4);
        z << static_cast<double>(detection.box.x) + detection.box.width / 2.0,
            static_cast<double>(detection.box.y) + detection.box.height / 2.0,
            static_cast<double>(detection.box.width), static_cast<double>(detection.box.height);
        if (has_box_model_) {
            box_kf_.update(z);
        } else {
            box_kf_ = makeConstantVelocityBox2D(detection.box);
            has_box_model_ = true;
        }
    }
    last_box_ = detection.box;
    class_id_ = detection.class_id;
    if (!detection.appearance.empty()) {
        if (appearance_.empty()) {
            appearance_ = detection.appearance.clone();
        } else {
            appearance_ =
                kAppearanceEma * appearance_ + (1.0 - kAppearanceEma) * detection.appearance;
        }
    }
}

void Track::markMissed() {
    ++time_since_update_;
    hit_streak_ = 0;
    if (reconfirm_after_miss_)
        confirmed_ = false;
}

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

#include "s3m/tracking/track.hpp"

namespace s3m {
namespace {

KalmanFilter::Vec toVec3(const cv::Point3f& p) {
    KalmanFilter::Vec v(3);
    v << static_cast<double>(p.x), static_cast<double>(p.y), static_cast<double>(p.z);
    return v;
}

}  // namespace

Track::Track(int id, const cv::Point3f& initial_position, double dt, double accel_std,
             double meas_std)
    : id_(id), kf_(makeConstantVelocity3D(dt, accel_std, meas_std)) {
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
    state.age = age_;
    state.hits = hits_;
    state.time_since_update = time_since_update_;
    state.confirmed = confirmed_;
    return state;
}

}  // namespace s3m

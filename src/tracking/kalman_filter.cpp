#include "s3m/tracking/kalman_filter.hpp"

#include <stdexcept>

namespace s3m {

KalmanFilter::KalmanFilter(int state_dim, int meas_dim) {
    if (state_dim <= 0 || meas_dim <= 0) {
        throw std::invalid_argument("KalmanFilter: state and measurement dims must be positive");
    }
    x_ = Vec::Zero(state_dim);
    P_ = Mat::Identity(state_dim, state_dim);
    F = Mat::Identity(state_dim, state_dim);
    H = Mat::Zero(meas_dim, state_dim);
    Q = Mat::Identity(state_dim, state_dim);
    R = Mat::Identity(meas_dim, meas_dim);
}

void KalmanFilter::init(const Vec& x0, const Mat& P0) {
    x_ = x0;
    P_ = P0;
}

void KalmanFilter::predict() {
    x_ = (F * x_).eval();  // explicit temporary: x_ appears on both sides
    P_ = F * P_ * F.transpose() + Q;
}

void KalmanFilter::update(const Vec& z) {
    const Vec innovation = z - H * x_;
    const Mat S = H * P_ * H.transpose() + R;
    const Mat K = P_ * H.transpose() * S.inverse();
    x_ += K * innovation;
    const Mat I = Mat::Identity(x_.size(), x_.size());
    P_ = (I - K * H) * P_;
}

double KalmanFilter::gatingDistanceSq(const Vec& z) const {
    const Vec innovation = z - H * x_;
    const Mat S = H * P_ * H.transpose() + R;
    return innovation.dot(S.ldlt().solve(innovation));
}

KalmanFilter makeConstantVelocity3D(double dt, double accel_std, double meas_std) {
    KalmanFilter kf(6, 3);

    KalmanFilter::Mat f = KalmanFilter::Mat::Identity(6, 6);
    f(0, 3) = dt;
    f(1, 4) = dt;
    f(2, 5) = dt;
    kf.F = f;

    KalmanFilter::Mat h = KalmanFilter::Mat::Zero(3, 6);
    h(0, 0) = 1.0;
    h(1, 1) = 1.0;
    h(2, 2) = 1.0;
    kf.H = h;

    // Discrete white-noise acceleration process noise.
    const double q = accel_std * accel_std;
    const double dt2 = dt * dt;
    const double dt3 = dt2 * dt;
    const double dt4 = dt2 * dt2;
    KalmanFilter::Mat process = KalmanFilter::Mat::Zero(6, 6);
    for (int i = 0; i < 3; ++i) {
        process(i, i) = dt4 / 4.0 * q;
        process(i, i + 3) = dt3 / 2.0 * q;
        process(i + 3, i) = dt3 / 2.0 * q;
        process(i + 3, i + 3) = dt2 * q;
    }
    kf.Q = process;

    kf.R = KalmanFilter::Mat::Identity(3, 3) * (meas_std * meas_std);

    KalmanFilter::Vec x0 = KalmanFilter::Vec::Zero(6);
    KalmanFilter::Mat p0 = KalmanFilter::Mat::Identity(6, 6);
    p0.block(0, 0, 3, 3) *= meas_std * meas_std;
    p0.block(3, 3, 3, 3) *= 10.0;  // velocity starts highly uncertain
    kf.init(x0, p0);

    return kf;
}

}  // namespace s3m

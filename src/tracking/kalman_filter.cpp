#include "s3m/tracking/kalman_filter.hpp"

#include <algorithm>
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

KalmanFilter makeConstantVelocityBox2D(const cv::Rect2f& initial_box) {
    KalmanFilter kf(8, 4);

    KalmanFilter::Mat f = KalmanFilter::Mat::Identity(8, 8);
    for (int i = 0; i < 4; ++i)
        f(i, i + 4) = 1.0;  // one step = one frame
    kf.F = f;

    KalmanFilter::Mat h = KalmanFilter::Mat::Zero(4, 8);
    for (int i = 0; i < 4; ++i)
        h(i, i) = 1.0;
    kf.H = h;

    KalmanFilter::Mat r = KalmanFilter::Mat::Identity(4, 4);
    r(2, 2) = r(3, 3) = 10.0;
    kf.R = r;

    KalmanFilter::Mat q = KalmanFilter::Mat::Identity(8, 8);
    for (int i = 4; i < 8; ++i)
        q(i, i) = 0.01;
    kf.Q = q;

    KalmanFilter::Vec x0 = KalmanFilter::Vec::Zero(8);
    x0(0) = static_cast<double>(initial_box.x) + initial_box.width / 2.0;
    x0(1) = static_cast<double>(initial_box.y) + initial_box.height / 2.0;
    x0(2) = static_cast<double>(initial_box.width);
    x0(3) = static_cast<double>(initial_box.height);
    KalmanFilter::Mat p0 = KalmanFilter::Mat::Identity(8, 8) * 10.0;
    for (int i = 4; i < 8; ++i)
        p0(i, i) *= 1000.0;
    kf.init(x0, p0);
    return kf;
}

KalmanFilter makeSortBoxFilter(const cv::Rect2f& initial_box) {
    KalmanFilter kf(7, 4);

    KalmanFilter::Mat f = KalmanFilter::Mat::Identity(7, 7);
    f(0, 4) = f(1, 5) = f(2, 6) = 1.0;
    kf.F = f;

    KalmanFilter::Mat h = KalmanFilter::Mat::Zero(4, 7);
    for (int i = 0; i < 4; ++i)
        h(i, i) = 1.0;
    kf.H = h;

    KalmanFilter::Mat r = KalmanFilter::Mat::Identity(4, 4);
    r(2, 2) = r(3, 3) = 10.0;
    kf.R = r;

    KalmanFilter::Mat q = KalmanFilter::Mat::Identity(7, 7);
    q(6, 6) *= 0.01;
    for (int i = 4; i < 7; ++i)
        q(i, i) *= 0.01;
    kf.Q = q;

    const double w = initial_box.width;
    const double hh = initial_box.height;
    KalmanFilter::Vec x0 = KalmanFilter::Vec::Zero(7);
    x0(0) = static_cast<double>(initial_box.x) + w / 2.0;
    x0(1) = static_cast<double>(initial_box.y) + hh / 2.0;
    x0(2) = w * hh;
    x0(3) = w / std::max(hh, 1e-9);
    KalmanFilter::Mat p0 = KalmanFilter::Mat::Identity(7, 7);
    for (int i = 4; i < 7; ++i)
        p0(i, i) *= 1000.0;
    p0 *= 10.0;
    kf.init(x0, p0);
    return kf;
}

}  // namespace s3m

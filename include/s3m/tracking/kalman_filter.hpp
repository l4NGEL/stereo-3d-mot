#pragma once

#include <Eigen/Dense>
#include <opencv2/core.hpp>

namespace s3m {

/// Generic linear Kalman filter (double precision, dynamic size).
///
///   predict:  x <- F x                P <- F P F^T + Q
///   update :  y  = z - H x            S = H P H^T + R
///             K  = P H^T S^-1         x <- x + K y      P <- (I - K H) P
///
/// The model matrices F, H, Q, R are public; set them directly or use a factory
/// such as makeConstantVelocity3D().
class KalmanFilter {
 public:
    using Vec = Eigen::VectorXd;
    using Mat = Eigen::MatrixXd;

    KalmanFilter() = default;
    KalmanFilter(int state_dim, int meas_dim);

    /// Set the initial state estimate and covariance.
    void init(const Vec& x0, const Mat& P0);

    void predict();
    void update(const Vec& z);

    /// Squared Mahalanobis distance between measurement `z` and the predicted
    /// measurement H x, using innovation covariance S. For gating.
    double gatingDistanceSq(const Vec& z) const;

    Mat F;  ///< state transition           (state x state)
    Mat H;  ///< measurement model          (meas  x state)
    Mat Q;  ///< process noise covariance   (state x state)
    Mat R;  ///< measurement noise covariance (meas x meas)

    const Vec& state() const { return x_; }
    const Mat& covariance() const { return P_; }
    Vec& mutableState() { return x_; }

    int stateDim() const { return static_cast<int>(x_.size()); }
    int measDim() const { return static_cast<int>(H.rows()); }

 private:
    Vec x_;  ///< state estimate
    Mat P_;  ///< state covariance
};

/// 3D constant-velocity filter.
///   state       = [x, y, z, vx, vy, vz]
///   measurement = [x, y, z]
///   dt          : timestep [s]
///   accel_std   : process noise as white acceleration [m/s^2]
///   meas_std    : position measurement noise [m]
KalmanFilter makeConstantVelocity3D(double dt, double accel_std, double meas_std);

/// Image-plane constant-velocity box filter, one step = one frame.
///   state       = [cx, cy, w, h, vcx, vcy, vw, vh]   (pixels, pixels/frame)
///   measurement = [cx, cy, w, h]
/// Noise constants follow the SORT reference implementation (Bewley et al.,
/// 2016 -- the same constants scripts/eval/sort_baseline.py uses), carried over
/// to a [cx, cy, w, h] parameterisation: R = diag(1, 1, 10, 10), P0 = 10 on the
/// box and 10000 on its velocity, Q = 1 on the box and 0.01 on its velocity.
/// They are declared, not tuned. Deliberately independent of stereo depth: the
/// 3D filter's projection would inherit the ~1.5 m object-depth error measured
/// in docs/evaluation/REPORT.md §3.
KalmanFilter makeConstantVelocityBox2D(const cv::Rect2f& initial_box);

}  // namespace s3m

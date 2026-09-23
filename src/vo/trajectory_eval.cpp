#include "s3m/vo/trajectory_eval.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include <opencv2/calib3d.hpp>

namespace s3m {
namespace {

/// Camera-to-world pose composed as 4x4 would be: T_a * T_b. Directly on
/// (R, t): (R_a*R_b, R_a*t_b + t_a).
Pose compose(const Pose& a, const Pose& b) {
    Pose out;
    out.rotation = a.rotation * b.rotation;
    const cv::Vec3d t_b(b.position.x, b.position.y, b.position.z);
    const cv::Vec3d t_a(a.position.x, a.position.y, a.position.z);
    const cv::Vec3d t = a.rotation * t_b + t_a;
    out.position = cv::Point3d(t[0], t[1], t[2]);
    return out;
}

Pose invert(const Pose& p) {
    Pose out;
    out.rotation = p.rotation.t();
    const cv::Vec3d t(p.position.x, p.position.y, p.position.z);
    const cv::Vec3d t_inv = -(out.rotation * t);
    out.position = cv::Point3d(t_inv[0], t_inv[1], t_inv[2]);
    return out;
}

double rotationAngleDeg(const cv::Matx33d& r) {
    const double tr = r(0, 0) + r(1, 1) + r(2, 2);
    const double c = std::clamp((tr - 1.0) / 2.0, -1.0, 1.0);
    return std::acos(c) * 180.0 / CV_PI;
}

}  // namespace

RigidAlignment alignRigid(const std::vector<cv::Point3d>& source, const std::vector<cv::Point3d>& target) {
    if (source.size() != target.size() || source.size() < 3) {
        throw std::invalid_argument("alignRigid: need matching point sets with at least 3 points");
    }
    const std::size_t n = source.size();

    cv::Point3d mean_s(0, 0, 0);
    cv::Point3d mean_t(0, 0, 0);
    for (std::size_t i = 0; i < n; ++i) {
        mean_s += source[i];
        mean_t += target[i];
    }
    mean_s *= 1.0 / static_cast<double>(n);
    mean_t *= 1.0 / static_cast<double>(n);

    // H = sum (source_i - mean_s) * (target_i - mean_t)^T  (3x3).
    cv::Matx33d H = cv::Matx33d::zeros();
    for (std::size_t i = 0; i < n; ++i) {
        const cv::Vec3d s = cv::Vec3d(source[i] - mean_s);
        const cv::Vec3d t = cv::Vec3d(target[i] - mean_t);
        H += s * t.t();
    }

    cv::Mat w;
    cv::Mat u;
    cv::Mat vt;
    cv::SVD::compute(cv::Mat(H), w, u, vt, cv::SVD::FULL_UV);
    cv::Matx33d U(u);
    cv::Matx33d Vt(vt);
    cv::Matx33d R = Vt.t() * U.t();
    if (cv::determinant(R) < 0.0) {
        // Reflection, not a rotation -- flip the sign of V's (equivalently
        // Vt's row) column paired with the smallest singular value.
        cv::Matx33d Vt_fixed = Vt;
        Vt_fixed(2, 0) *= -1;
        Vt_fixed(2, 1) *= -1;
        Vt_fixed(2, 2) *= -1;
        R = Vt_fixed.t() * U.t();
    }

    const cv::Vec3d t_align = cv::Vec3d(mean_t) - R * cv::Vec3d(mean_s);
    RigidAlignment out;
    out.rotation = R;
    out.translation = cv::Point3d(t_align[0], t_align[1], t_align[2]);
    return out;
}

TrajectoryError evaluateTrajectory(const std::vector<Pose>& estimated,
                                   const std::vector<Pose>& ground_truth, int delta) {
    if (estimated.size() != ground_truth.size()) {
        throw std::invalid_argument("evaluateTrajectory: estimated/ground_truth size mismatch");
    }
    TrajectoryError err;
    err.num_poses = static_cast<int>(estimated.size());
    if (estimated.empty()) return err;

    std::vector<cv::Point3d> est_pos;
    std::vector<cv::Point3d> gt_pos;
    est_pos.reserve(estimated.size());
    gt_pos.reserve(estimated.size());
    for (const Pose& p : estimated) est_pos.push_back(p.position);
    for (const Pose& p : ground_truth) gt_pos.push_back(p.position);

    const RigidAlignment align = alignRigid(est_pos, gt_pos);
    double sq_sum = 0.0;
    double abs_sum = 0.0;
    for (std::size_t i = 0; i < est_pos.size(); ++i) {
        const cv::Vec3d aligned = align.rotation * cv::Vec3d(est_pos[i]) + cv::Vec3d(align.translation);
        const double e = cv::norm(aligned - cv::Vec3d(gt_pos[i]));
        sq_sum += e * e;
        abs_sum += e;
    }
    err.ate_rmse = std::sqrt(sq_sum / static_cast<double>(est_pos.size()));
    err.ate_mean = abs_sum / static_cast<double>(est_pos.size());

    double rpe_sq_trans = 0.0;
    double rpe_sq_rot = 0.0;
    int pairs = 0;
    for (std::size_t i = 0; i + static_cast<std::size_t>(delta) < estimated.size(); ++i) {
        const std::size_t j = i + static_cast<std::size_t>(delta);
        const Pose q_rel = compose(invert(ground_truth[j]), ground_truth[i]);
        const Pose p_rel = compose(invert(estimated[j]), estimated[i]);
        const Pose e_rel = compose(invert(q_rel), p_rel);

        const double trans_err = cv::norm(cv::Vec3d(e_rel.position));
        const double rot_err_deg = rotationAngleDeg(e_rel.rotation);
        rpe_sq_trans += trans_err * trans_err;
        rpe_sq_rot += rot_err_deg * rot_err_deg;
        ++pairs;
    }
    err.rpe_num_pairs = pairs;
    if (pairs > 0) {
        err.rpe_trans_rmse = std::sqrt(rpe_sq_trans / pairs);
        err.rpe_rot_rmse_deg = std::sqrt(rpe_sq_rot / pairs);
    }
    return err;
}

}  // namespace s3m

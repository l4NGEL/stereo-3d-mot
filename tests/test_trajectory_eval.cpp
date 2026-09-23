#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include "s3m/vo/trajectory_eval.hpp"

using namespace s3m;

namespace {

Pose makePose(const cv::Matx33d& r, cv::Point3d t) {
    Pose p;
    p.rotation = r;
    p.position = t;
    return p;
}

cv::Matx33d rotateZ(double deg) {
    const double a = deg * CV_PI / 180.0;
    return cv::Matx33d(std::cos(a), -std::sin(a), 0, std::sin(a), std::cos(a), 0, 0, 0, 1);
}

}  // namespace

// A non-degenerate (full 3D spread) point cloud, a known rotation+translation
// applied to it, then aligned back -- mirrors the Python validation this
// module's design was checked against before porting (see docs/roadmap.md
// Phase 7) but for the alignment step specifically, using OpenCV types
// directly rather than re-deriving in Python: Kabsch/Umeyama is textbook,
// well-established algebra, not a formula this project invented, so a
// hand-verified C++ case (rather than an external Python cross-check) is the
// appropriate level of rigor here.
TEST(AlignRigid, RecoversKnownRotationAndTranslation) {
    const std::vector<cv::Point3d> source = {
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 1}, {2, -1, 0.5},
    };
    const cv::Matx33d R_true = rotateZ(30.0);
    const cv::Point3d t_true(5.0, -2.0, 1.0);

    std::vector<cv::Point3d> target(source.size());
    for (std::size_t i = 0; i < source.size(); ++i) {
        const cv::Vec3d s(source[i]);
        const cv::Vec3d r = R_true * s + cv::Vec3d(t_true);
        target[i] = cv::Point3d(r[0], r[1], r[2]);
    }

    const RigidAlignment align = alignRigid(source, target);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) EXPECT_NEAR(align.rotation(r, c), R_true(r, c), 1e-9);
    EXPECT_NEAR(align.translation.x, t_true.x, 1e-9);
    EXPECT_NEAR(align.translation.y, t_true.y, 1e-9);
    EXPECT_NEAR(align.translation.z, t_true.z, 1e-9);
}

TEST(AlignRigid, ThrowsOnMismatchedOrTooFewPoints) {
    const std::vector<cv::Point3d> three = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    const std::vector<cv::Point3d> two = {{0, 0, 0}, {1, 0, 0}};
    EXPECT_THROW(alignRigid(two, two), std::invalid_argument);       // < 3 points
    EXPECT_THROW(alignRigid(three, two), std::invalid_argument);     // size mismatch
}

TEST(EvaluateTrajectory, ZeroErrorForIdenticalTrajectories) {
    std::vector<Pose> traj = {
        makePose(cv::Matx33d::eye(), {0, 0, 0}),
        makePose(rotateZ(5.0), {1, 0, 0}),
        makePose(rotateZ(12.0), {2, 0.2, 0}),
        makePose(rotateZ(20.0), {3, 0.5, 0.1}),
    };
    const TrajectoryError err = evaluateTrajectory(traj, traj, /*delta=*/1);
    EXPECT_NEAR(err.ate_rmse, 0.0, 1e-9);
    EXPECT_NEAR(err.ate_mean, 0.0, 1e-9);
    EXPECT_NEAR(err.rpe_trans_rmse, 0.0, 1e-9);
    EXPECT_NEAR(err.rpe_rot_rmse_deg, 0.0, 1e-9);
    EXPECT_EQ(err.num_poses, 4);
    EXPECT_EQ(err.rpe_num_pairs, 3);
}

// Textbook property, of BOTH metrics: applying the same fixed rigid
// transform G to every pose in a trajectory (a common "gauge" shift of its
// arbitrary global reference frame) changes neither. RPE: G cancels
// directly in Pose[i]^-1 * Pose[i+1] regardless of what G is. ATE: less
// obvious but equally exact -- alignRigid finds the *optimal* rigid
// alignment, and since est == G * gt exactly here, that optimum is G^-1,
// which undoes the shift perfectly, leaving zero residual. (An earlier
// version of this test wrongly expected ATE to blow up under the gauge
// shift -- it doesn't, and that's alignment doing exactly its job, not a
// bug: this is why ATE needs an alignment step and raw position error
// wouldn't do.)
TEST(EvaluateTrajectory, InvariantToAGlobalGaugeTransform) {
    const std::vector<Pose> gt = {
        makePose(cv::Matx33d::eye(), {0, 0, 0}),
        makePose(rotateZ(5.0), {1, 0, 0}),
        makePose(rotateZ(12.0), {2, 0.2, 0}),
    };

    const Pose G = makePose(rotateZ(47.0), {100, -50, 3});
    std::vector<Pose> est(gt.size());
    for (std::size_t i = 0; i < gt.size(); ++i) {
        Pose p;
        p.rotation = G.rotation * gt[i].rotation;
        const cv::Vec3d t = G.rotation * cv::Vec3d(gt[i].position) + cv::Vec3d(G.position);
        p.position = cv::Point3d(t[0], t[1], t[2]);
        est[i] = p;
    }

    const TrajectoryError err = evaluateTrajectory(est, gt, /*delta=*/1);
    EXPECT_NEAR(err.rpe_trans_rmse, 0.0, 1e-9);
    EXPECT_NEAR(err.rpe_rot_rmse_deg, 0.0, 1e-9);
    EXPECT_NEAR(err.ate_rmse, 0.0, 1e-9);

    // G was not a no-op: confirm the RAW (pre-alignment) position gap this
    // test actually shifted by really is large, so a zero ATE above is
    // alignment correctly undoing a real shift, not a vacuously-true check.
    double raw_sq = 0.0;
    for (std::size_t i = 0; i < gt.size(); ++i) {
        const cv::Vec3d d = cv::Vec3d(est[i].position) - cv::Vec3d(gt[i].position);
        raw_sq += d.dot(d);
    }
    EXPECT_GT(std::sqrt(raw_sq / static_cast<double>(gt.size())), 10.0);
}

// Alignment can only ever reduce (or match) the raw, unaligned RMSE -- it's
// an optimum over all rigid transforms, and the identity transform is one of
// them. A perturbation that's zero-mean and not itself a rigid transform of
// the trajectory gives a clean, always-true upper bound without needing to
// hand-solve the optimal alignment.
TEST(EvaluateTrajectory, AlignedAteNeverExceedsRawRmse) {
    const std::vector<Pose> gt = {
        makePose(cv::Matx33d::eye(), {0, 0, 0}),
        makePose(cv::Matx33d::eye(), {1, 0, 0}),
        makePose(cv::Matx33d::eye(), {2, 0, 0}),
        makePose(cv::Matx33d::eye(), {3, 0, 0}),
    };
    const double perturb[] = {0.1, -0.1, 0.15, -0.15};
    std::vector<Pose> est = gt;
    double raw_sq = 0.0;
    for (std::size_t i = 0; i < est.size(); ++i) {
        est[i].position.y += perturb[i];
        raw_sq += perturb[i] * perturb[i];
    }
    const double raw_rmse = std::sqrt(raw_sq / static_cast<double>(est.size()));

    const TrajectoryError err = evaluateTrajectory(est, gt, /*delta=*/1);
    EXPECT_GT(err.ate_rmse, 0.0);
    EXPECT_LE(err.ate_rmse, raw_rmse + 1e-9);
}

TEST(EvaluateTrajectory, ThrowsOnSizeMismatch) {
    const std::vector<Pose> one = {makePose(cv::Matx33d::eye(), {0, 0, 0})};
    const std::vector<Pose> two = {makePose(cv::Matx33d::eye(), {0, 0, 0}),
                                   makePose(cv::Matx33d::eye(), {1, 0, 0})};
    EXPECT_THROW(evaluateTrajectory(one, two), std::invalid_argument);
}

// Tests for the predicted 2D box (Track::predictedBox) and AssociationMethod::kTwoStage.
// The scenario these guard against is the one docs/evaluation/IDSW_REPORT.md measured
// on held-out KITTI: a track that has already lost its object (coasting, frozen last
// box) out-competing the active track for that object's detection.
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include "s3m/camera/stereo_rig.hpp"
#include "s3m/tracking/appearance.hpp"
#include "s3m/tracking/track.hpp"
#include "s3m/tracking/tracker.hpp"

using namespace s3m;

namespace {

const StereoRig& testRig() {
    static const StereoRig rig =
        StereoRig::fromIntrinsics(600.0, 600.0, 320.0, 240.0, cv::Size(640, 480), 0.12);
    return rig;
}

Detection3D det(cv::Rect2f box, double depth, int class_id = 0) {
    const cv::Point2d center(box.x + box.width / 2.0, box.y + box.height / 2.0);
    const cv::Point3d p = testRig().left().backProject(center, depth);
    Detection3D d;
    d.box = box;
    d.position =
        cv::Point3f(static_cast<float>(p.x), static_cast<float>(p.y), static_cast<float>(p.z));
    d.depth = static_cast<float>(depth);
    d.class_id = class_id;
    d.score = 0.9f;
    d.valid = true;
    return d;
}

Detection3D coloured(cv::Rect2f box, double depth, cv::Scalar bgr) {
    Detection3D d = det(box, depth);
    const cv::Mat crop(40, 40, CV_8UC3, bgr);
    d.appearance = computeAppearanceDescriptor(crop, cv::Rect2f(0, 0, 40, 40));
    return d;
}

const cv::Scalar kRed(0, 0, 255);
const cv::Scalar kBlue(255, 0, 0);

TrackerParams twoStage(LostTrackCost lost = LostTrackCost::kAppearanceMahalanobis) {
    TrackerParams p;
    p.dt = 0.1;
    p.process_noise = 1.0;
    p.measurement_noise = 0.05;
    p.max_age = 5;
    p.min_hits = 3;
    p.association = AssociationMethod::kTwoStage;
    p.iou_gate = 0.3;
    p.lost_cost = lost;
    p.lost_gating_chi2 = 7.815;
    p.lost_max_appearance = 0.5;
    p.collect_diagnostics = true;
    return p;
}

int stageOf(const TrackerUpdateResult& r, int track_id) {
    for (const AssociationDiagnostic& g : r.diagnostics)
        if (g.track_id == track_id && g.matched_stage > 0)
            return g.matched_stage;
    return 0;
}

// Two objects A (x=100) and B (x=125) at the same depth; B then disappears for two
// frames, and a single detection appears at x=118 -- A's next observation, but it
// overlaps B's frozen box (IoU 0.70) more than A's (IoU 0.38, still above the 0.3
// gate). Returns {id of A, id of B, id the detection got}.
struct StealResult {
    int a = -1, b = -1, got = -1;
};
StealResult runStealScenario(const TrackerParams& p) {
    Tracker tracker(p);
    const cv::Rect2f box_a(100, 100, 40, 40), box_b(125, 100, 40, 40);
    TrackerUpdateResult r;
    for (int i = 0; i < 3; ++i)
        r = tracker.update({det(box_a, 5.0), det(box_b, 5.0)});
    StealResult s;
    s.a = r.detection_track_id[0];
    s.b = r.detection_track_id[1];
    // B disappears; A keeps being seen where it is.
    tracker.update({det(box_a, 5.0)});
    tracker.update({det(box_a, 5.0)});
    const cv::Rect2f moved(118, 100, 40, 40);
    r = tracker.update({det(moved, 5.0)});
    s.got = r.detection_track_id[0];
    return s;
}

}  // namespace

// --- predicted box -------------------------------------------------------------------

TEST(TrackPredictedBox, AdvancesExactlyOneStepPerTrackerFrame) {
    TrackerParams p = twoStage();
    Tracker tracker(p);
    // Constant +10 px/frame motion.
    for (int i = 0; i < 8; ++i)
        tracker.update({det(cv::Rect2f(100.0f + 10.0f * i, 100, 40, 40), 5.0)});
    ASSERT_EQ(tracker.tracks().size(), 1u);
    const cv::Rect2f last = tracker.tracks()[0].lastBox();  // x = 170

    tracker.update({});  // one missed frame -> exactly one predict() happened in it
    ASSERT_EQ(tracker.tracks().size(), 1u);
    const cv::Rect2f pred = tracker.tracks()[0].predictedBox();
    EXPECT_NEAR(pred.x, last.x + 10.0f, 2.0f);  // one step, not two (would be +20)
    EXPECT_NEAR(pred.width, 40.0f, 1.0f);
    EXPECT_EQ(tracker.tracks()[0].lastBox(), last);  // observed box is kept separately
}

TEST(TrackPredictedBox, ShrinkingBoxNeverGoesBelowOnePixelOrNonFinite) {
    Track t(1, cv::Point3f(0, 0, 5), 0.1, 1.0, 0.05, 0, cv::Rect2f(100, 100, 60, 60));
    for (int i = 1; i < 6; ++i) {
        t.predict();
        const float w = 60.0f - 10.0f * i;
        t.correct(det(cv::Rect2f(100, 100, w, w), 5.0));
    }
    for (int i = 0; i < 20; ++i) {
        t.predict();
        const cv::Rect2f b = t.predictedBox();
        ASSERT_TRUE(std::isfinite(b.x) && std::isfinite(b.y));
        EXPECT_GE(b.width, 1.0f);
        EXPECT_GE(b.height, 1.0f);
    }
}

TEST(TrackPredictedBox, NoBoxModelFallsBackToLastBox) {
    Track t(1, cv::Point3f(0, 0, 5), 0.1, 1.0, 0.05);  // born without a box
    t.predict();
    EXPECT_EQ(t.predictedBox(), t.lastBox());
}

TEST(TrackPredictedBox, BaselineIou2DIgnoresPredictionByDefault) {
    TrackerParams p;
    EXPECT_FALSE(p.iou_use_predicted_box);  // the evaluated baseline is unchanged
    EXPECT_EQ(p.association, AssociationMethod::kMahalanobis3D);
}

// --- the measured failure mode, and the two-stage fix ----------------------------------

TEST(TwoStage, FrozenLostBoxStealsDetectionUnderBaselineIou2D) {
    // Documents the mechanism: with last-box IoU, the lost track B wins the detection
    // that is really A's, because B's frozen box overlaps it more.
    TrackerParams p = twoStage();
    p.association = AssociationMethod::kIou2D;
    const StealResult s = runStealScenario(p);
    ASSERT_NE(s.a, s.b);
    EXPECT_EQ(s.got, s.b);
}

TEST(TwoStage, LostTrackCannotTakeDetectionFromActiveTrack) {
    // Same scene; stage 1 gives A first claim. lost_cost = IoU so B would *want* it.
    const StealResult s = runStealScenario(twoStage(LostTrackCost::kIouPredicted));
    ASSERT_NE(s.a, s.b);
    EXPECT_EQ(s.got, s.a);
}

TEST(TwoStage, StageOneMatchIsNotRevisitedByStageTwo) {
    Tracker tracker(twoStage(LostTrackCost::kIouPredicted));
    const cv::Rect2f box(100, 100, 40, 40);
    TrackerUpdateResult r;
    for (int i = 0; i < 3; ++i)
        r = tracker.update({det(box, 5.0), det(cv::Rect2f(105, 100, 40, 40), 5.0)});
    const int a = r.detection_track_id[0];
    const int b = r.detection_track_id[1];
    tracker.update({det(box, 5.0)});  // b now lost
    r = tracker.update({det(box, 5.0)});
    EXPECT_EQ(r.detection_track_id[0], a);
    EXPECT_EQ(stageOf(r, a), 1);
    EXPECT_EQ(stageOf(r, b), 0);  // one detection, one match, per frame
}

TEST(TwoStage, RecoversSameIdentityAfterShortGap) {
    Tracker tracker(twoStage());
    const cv::Rect2f box(300, 200, 40, 40);
    TrackerUpdateResult r;
    for (int i = 0; i < 3; ++i)
        r = tracker.update({coloured(box, 5.0, kRed)});
    const int id = r.detection_track_id[0];
    for (int i = 0; i < 3; ++i)
        tracker.update({});
    r = tracker.update({coloured(box, 5.0, kRed)});
    EXPECT_EQ(r.detection_track_id[0], id);
    EXPECT_EQ(stageOf(r, id), 2);
    ASSERT_EQ(r.tracks.size(), 1u);
    EXPECT_TRUE(isEmittable(r.tracks[0]));
}

TEST(TwoStage, RejectsRecoveryWithDifferentAppearance) {
    Tracker tracker(twoStage());
    const cv::Rect2f box(300, 200, 40, 40);
    TrackerUpdateResult r;
    for (int i = 0; i < 3; ++i)
        r = tracker.update({coloured(box, 5.0, kRed)});
    const int id = r.detection_track_id[0];
    tracker.update({});
    r = tracker.update({coloured(box, 5.0, kBlue)});  // same place, other colour
    EXPECT_NE(r.detection_track_id[0], id);
}

TEST(TwoStage, RejectsSimilarLookingObjectAtDifferentDepth) {
    // Same colour, same image box, 4 m further away: the Mahalanobis gate refuses it.
    Tracker tracker(twoStage());
    const cv::Rect2f box(300, 200, 40, 40);
    TrackerUpdateResult r;
    for (int i = 0; i < 3; ++i)
        r = tracker.update({coloured(box, 5.0, kRed)});
    const int id = r.detection_track_id[0];
    tracker.update({});
    r = tracker.update({coloured(box, 9.0, kRed)});
    EXPECT_NE(r.detection_track_id[0], id);
}

TEST(TwoStage, MissingDescriptorIsNotALooserMatch) {
    Tracker tracker(twoStage());
    const cv::Rect2f box(300, 200, 40, 40);
    TrackerUpdateResult r;
    for (int i = 0; i < 3; ++i)
        r = tracker.update({coloured(box, 5.0, kRed)});
    const int id = r.detection_track_id[0];
    tracker.update({});
    r = tracker.update({det(box, 5.0)});  // no appearance descriptor
    EXPECT_NE(r.detection_track_id[0], id);
}

TEST(TwoStage, LostCostNoneNeverRecovers) {
    Tracker tracker(twoStage(LostTrackCost::kNone));
    const cv::Rect2f box(300, 200, 40, 40);
    TrackerUpdateResult r;
    for (int i = 0; i < 3; ++i)
        r = tracker.update({coloured(box, 5.0, kRed)});
    const int id = r.detection_track_id[0];
    tracker.update({});
    r = tracker.update({coloured(box, 5.0, kRed)});
    EXPECT_NE(r.detection_track_id[0], id);
}

TEST(TwoStage, EmptyAndInvalidInputs) {
    Tracker tracker(twoStage());
    TrackerUpdateResult r = tracker.update({});
    EXPECT_TRUE(r.tracks.empty());
    Detection3D bad = coloured(cv::Rect2f(300, 200, 40, 40), 5.0, kRed);
    bad.valid = false;  // no depth: never enters the tracker
    r = tracker.update({bad});
    EXPECT_TRUE(r.tracks.empty());
    EXPECT_EQ(r.detection_track_id[0], -1);
    // A lost track facing only an invalid detection stays lost, no crash.
    for (int i = 0; i < 3; ++i)
        tracker.update({coloured(cv::Rect2f(300, 200, 40, 40), 5.0, kRed)});
    r = tracker.update({bad});
    ASSERT_EQ(r.tracks.size(), 1u);
    EXPECT_EQ(r.tracks[0].time_since_update, 1);
}

TEST(TwoStage, MaxAgeBoundary) {
    TrackerParams p = twoStage();
    p.max_age = 2;
    const cv::Rect2f box(300, 200, 40, 40);
    {
        Tracker tracker(p);
        TrackerUpdateResult r;
        for (int i = 0; i < 3; ++i)
            r = tracker.update({coloured(box, 5.0, kRed)});
        const int id = r.detection_track_id[0];
        tracker.update({});
        tracker.update({});  // tsu == max_age: kept
        r = tracker.update({coloured(box, 5.0, kRed)});
        EXPECT_EQ(r.detection_track_id[0], id);
    }
    {
        Tracker tracker(p);
        TrackerUpdateResult r;
        for (int i = 0; i < 3; ++i)
            r = tracker.update({coloured(box, 5.0, kRed)});
        const int id = r.detection_track_id[0];
        for (int i = 0; i < 3; ++i)
            tracker.update({});  // tsu 3 > max_age: pruned
        EXPECT_TRUE(tracker.tracks().empty());
        r = tracker.update({coloured(box, 5.0, kRed)});
        EXPECT_NE(r.detection_track_id[0], id);
    }
}

TEST(TwoStage, TentativeLifecycleAndEmission) {
    Tracker tracker(twoStage());
    const cv::Rect2f box(300, 200, 40, 40);
    TrackerUpdateResult r = tracker.update({coloured(box, 5.0, kRed)});
    ASSERT_EQ(r.tracks.size(), 1u);
    EXPECT_FALSE(r.tracks[0].confirmed);
    EXPECT_FALSE(isEmittable(r.tracks[0]));
    const int id = r.tracks[0].id;
    r = tracker.update({coloured(box, 5.0, kRed)});  // tentative, active -> stage 1
    EXPECT_EQ(stageOf(r, id), 1);
    r = tracker.update({coloured(box, 5.0, kRed)});  // 3rd hit -> confirmed
    EXPECT_TRUE(r.tracks[0].confirmed);
    EXPECT_TRUE(isEmittable(r.tracks[0]));
    r = tracker.update({});  // coasting: confirmed but not emitted (E = 0 unchanged)
    EXPECT_TRUE(r.tracks[0].confirmed);
    EXPECT_FALSE(isEmittable(r.tracks[0]));
}

TEST(TwoStage, FromConfigMapsNewKeys) {
    TrackingParams tp;
    tp.association = "two_stage";
    tp.stage1_max_tsu = 1;
    tp.lost_cost = "iou";
    tp.lost_gating_chi2 = 5.0;
    tp.lost_max_appearance = 0.3;
    TrackerParams p = TrackerParams::fromConfig(tp);
    EXPECT_EQ(p.association, AssociationMethod::kTwoStage);
    EXPECT_EQ(p.stage1_max_tsu, 1);
    EXPECT_EQ(p.lost_cost, LostTrackCost::kIouPredicted);
    EXPECT_DOUBLE_EQ(p.lost_gating_chi2, 5.0);
    EXPECT_DOUBLE_EQ(p.lost_max_appearance, 0.3);
    tp.lost_cost = "none";
    EXPECT_EQ(TrackerParams::fromConfig(tp).lost_cost, LostTrackCost::kNone);
    tp.association = "iou2d";
    tp.iou_use_predicted_box = true;
    p = TrackerParams::fromConfig(tp);
    EXPECT_EQ(p.association, AssociationMethod::kIou2D);
    EXPECT_TRUE(p.iou_use_predicted_box);
}

// --- SORT's reporting rule (reconfirm_after_miss) -------------------------------------

TEST(Reconfirm, MissRevokesEmissionUntilMinHitsConsecutiveMatches) {
    TrackerParams p = twoStage();
    p.association = AssociationMethod::kIou2D;
    p.iou_use_predicted_box = true;
    p.reconfirm_after_miss = true;
    Tracker tracker(p);
    const cv::Rect2f box(300, 200, 40, 40);
    TrackerUpdateResult r;
    for (int i = 0; i < 3; ++i)
        r = tracker.update({det(box, 5.0)});
    ASSERT_TRUE(isEmittable(r.tracks[0]));
    const int id = r.tracks[0].id;

    r = tracker.update({});  // miss: confirmation revoked, track kept
    ASSERT_EQ(r.tracks.size(), 1u);
    EXPECT_FALSE(r.tracks[0].confirmed);

    r = tracker.update({det(box, 5.0)});  // re-matched, same id, streak 1
    EXPECT_EQ(r.tracks[0].id, id);
    EXPECT_FALSE(isEmittable(r.tracks[0]));
    r = tracker.update({det(box, 5.0)});  // streak 2
    EXPECT_FALSE(isEmittable(r.tracks[0]));
    r = tracker.update({det(box, 5.0)});  // streak 3 == min_hits
    EXPECT_TRUE(isEmittable(r.tracks[0]));
    EXPECT_EQ(r.tracks[0].id, id);
}

TEST(Reconfirm, DefaultKeepsConfirmationAcrossMisses) {
    TrackerParams p = twoStage();
    p.association = AssociationMethod::kIou2D;
    EXPECT_FALSE(p.reconfirm_after_miss);
    Tracker tracker(p);
    const cv::Rect2f box(300, 200, 40, 40);
    TrackerUpdateResult r;
    for (int i = 0; i < 3; ++i)
        r = tracker.update({det(box, 5.0)});
    tracker.update({});
    r = tracker.update({det(box, 5.0)});
    EXPECT_TRUE(isEmittable(r.tracks[0]));  // baseline: emitted again immediately
}

TEST(Reconfirm, FromConfig) {
    TrackingParams tp;
    tp.reconfirm_after_miss = true;
    EXPECT_TRUE(TrackerParams::fromConfig(tp).reconfirm_after_miss);
}

// --- SORT box filter and SORT assignment objective -------------------------------------

TEST(SortBoxModel, AdvancesOneStepAndStaysValid) {
    TrackerParams p = twoStage();
    p.association = AssociationMethod::kIou2D;
    p.iou_use_predicted_box = true;
    p.sort_box_model = true;
    Tracker tracker(p);
    for (int i = 0; i < 8; ++i)
        tracker.update({det(cv::Rect2f(100.0f + 10.0f * i, 100, 40, 40), 5.0)});
    const cv::Rect2f last = tracker.tracks()[0].lastBox();
    tracker.update({});
    const cv::Rect2f pred = tracker.tracks()[0].predictedBox();
    EXPECT_NEAR(pred.x, last.x + 10.0f, 2.0f);
    EXPECT_NEAR(pred.width, 40.0f, 1.0f);
    EXPECT_NEAR(pred.height, 40.0f, 1.0f);
}

TEST(SortBoxModel, ShrinkingAreaNeverInvalid) {
    Track t(1, cv::Point3f(0, 0, 5), 0.1, 1.0, 0.05, 0, cv::Rect2f(100, 100, 60, 60));
    t.setSortBoxModel(true);
    for (int i = 1; i < 6; ++i) {
        t.predict();
        const float w = 60.0f - 10.0f * i;
        t.correct(det(cv::Rect2f(100, 100, w, w), 5.0));
    }
    for (int i = 0; i < 20; ++i) {
        t.predict();
        const cv::Rect2f b = t.predictedBox();
        ASSERT_TRUE(std::isfinite(b.x) && std::isfinite(b.width));
        EXPECT_GE(b.width, 1.0f);
        EXPECT_GE(b.height, 1.0f);
    }
}

// T1 (x=100) and T2 (x=121) exist; next frame D1 (x=102) overlaps T1 strongly (0.90)
// and T2 weakly (0.36), D2 (x=79) overlaps only T1, weakly (0.31). Cardinality-first
// matches both weak pairs (T1-D2, T2-D1); SORT's max-total-IoU keeps T1-D1.
static std::vector<int> runAssignmentScene(bool max_total) {
    TrackerParams p = twoStage();
    p.association = AssociationMethod::kIou2D;
    p.iou_max_total_assignment = max_total;
    Tracker tracker(p);
    TrackerUpdateResult r;
    for (int i = 0; i < 3; ++i)
        r = tracker.update(
            {det(cv::Rect2f(100, 100, 40, 40), 5.0), det(cv::Rect2f(121, 100, 40, 40), 5.0)});
    const int t1 = r.detection_track_id[0];
    const int t2 = r.detection_track_id[1];
    r = tracker.update(
        {det(cv::Rect2f(102, 100, 40, 40), 5.0), det(cv::Rect2f(79, 100, 40, 40), 5.0)});
    return {t1, t2, r.detection_track_id[0], r.detection_track_id[1]};
}

TEST(IouAssignment, DefaultPrefersMorePairs) {
    const std::vector<int> ids = runAssignmentScene(false);
    EXPECT_EQ(ids[2], ids[1]);  // D1 -> T2
    EXPECT_EQ(ids[3], ids[0]);  // D2 -> T1
}

TEST(IouAssignment, MaxTotalKeepsStrongPair) {
    const std::vector<int> ids = runAssignmentScene(true);
    EXPECT_EQ(ids[2], ids[0]);  // D1 -> T1
    EXPECT_NE(ids[3], ids[0]);  // D2 -> not T1 (T2's IoU 0 is dropped by the gate)
    EXPECT_NE(ids[3], ids[1]);
}

// --- round 4 options --------------------------------------------------------------------

namespace {
TrackerParams predIou() {
    TrackerParams p = twoStage();
    p.association = AssociationMethod::kIou2D;
    p.iou_use_predicted_box = true;
    p.max_age = 30;
    return p;
}
}  // namespace

TEST(BoxCoast, PredictedBoxStopsAfterKUnmatchedFrames) {
    for (const int k : {0, 1}) {
        TrackerParams p = predIou();
        p.box_coast_frames = k;
        Tracker tracker(p);
        for (int i = 0; i < 8; ++i)
            tracker.update({det(cv::Rect2f(100.0f + 10.0f * i, 100, 40, 40), 5.0)});
        const float last_x = tracker.tracks()[0].lastBox().x;  // 170
        for (int i = 0; i < 5; ++i)
            tracker.update({});
        const float px = tracker.tracks()[0].predictedBox().x;
        if (k == 0)
            EXPECT_NEAR(px, last_x + 50.0f, 4.0f);  // keeps extrapolating
        else
            EXPECT_NEAR(px, last_x + 10.0f, 3.0f);  // one step, then frozen
    }
}

TEST(BoxCoast, MovingAgainAfterRematch) {
    TrackerParams p = predIou();
    p.box_coast_frames = 1;
    Tracker tracker(p);
    for (int i = 0; i < 8; ++i)
        tracker.update({det(cv::Rect2f(100.0f + 10.0f * i, 100, 40, 40), 5.0)});
    tracker.update({});
    tracker.update({});
    // Re-matched near where it stopped: the track is tracked again (same id).
    const TrackerUpdateResult r = tracker.update({det(cv::Rect2f(185, 100, 40, 40), 5.0)});
    ASSERT_EQ(r.tracks.size(), 1u);
    EXPECT_EQ(r.tracks[0].time_since_update, 0);
}

TEST(FilteredBox, SnapshotReportsFilterEstimateOnlyWhenEnabled) {
    for (const bool on : {false, true}) {
        TrackerParams p = predIou();
        p.report_filtered_box = on;
        Tracker tracker(p);
        TrackerUpdateResult r;
        for (int i = 0; i < 4; ++i)
            r = tracker.update({det(cv::Rect2f(100, 100, 40, 40), 5.0)});
        const cv::Rect2f jumped(112, 100, 40, 40);  // a noisy detection
        r = tracker.update({det(jumped, 5.0)});
        ASSERT_EQ(r.tracks.size(), 1u);
        if (on) {
            EXPECT_LT(r.tracks[0].box.x, 112.0f);  // smoothed towards the track
            EXPECT_GT(r.tracks[0].box.x, 100.0f);
            EXPECT_GE(r.tracks[0].box.width, 1.0f);
        } else {
            EXPECT_EQ(r.tracks[0].box, jumped);  // baseline: raw detection box
        }
        EXPECT_EQ(tracker.tracks()[0].lastBox(), jumped);  // observed box kept either way
    }
}

TEST(SortWarmup, TracksInFirstMinHitsFramesAreEmittedAtOnce) {
    TrackerParams p = predIou();
    p.sort_warmup = true;  // min_hits = 3
    Tracker tracker(p);
    TrackerUpdateResult r = tracker.update({det(cv::Rect2f(100, 100, 40, 40), 5.0)});
    ASSERT_EQ(r.tracks.size(), 1u);
    EXPECT_TRUE(isEmittable(r.tracks[0]));  // frame 1: born and emitted
    tracker.update({det(cv::Rect2f(100, 100, 40, 40), 5.0)});
    tracker.update({det(cv::Rect2f(100, 100, 40, 40), 5.0)});
    // frame 4: a newly born track is tentative again, as usual
    r = tracker.update(
        {det(cv::Rect2f(100, 100, 40, 40), 5.0), det(cv::Rect2f(400, 100, 40, 40), 5.0)});
    ASSERT_EQ(r.tracks.size(), 2u);
    EXPECT_FALSE(isEmittable(r.tracks[1]));
}

TEST(SortWarmup, OffByDefaultAndResetRestartsTheWindow) {
    TrackerParams p = predIou();
    EXPECT_FALSE(p.sort_warmup);
    EXPECT_EQ(p.box_coast_frames, 0);
    EXPECT_FALSE(p.report_filtered_box);
    p.sort_warmup = true;
    Tracker tracker(p);
    for (int i = 0; i < 5; ++i)
        tracker.update({});
    tracker.reset();
    const TrackerUpdateResult r = tracker.update({det(cv::Rect2f(100, 100, 40, 40), 5.0)});
    EXPECT_TRUE(isEmittable(r.tracks[0]));
}

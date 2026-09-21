#include <vector>

#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include "s3m/camera/stereo_rig.hpp"
#include "s3m/tracking/appearance.hpp"
#include "s3m/tracking/tracker.hpp"

using namespace s3m;

namespace {

const StereoRig& testRig() {
    static const StereoRig rig =
        StereoRig::fromIntrinsics(600.0, 600.0, 320.0, 240.0, cv::Size(640, 480), 0.12);
    return rig;
}

Detection3D makeDetection(cv::Rect2f box, double depth, int class_id = 0, float score = 0.9f) {
    const cv::Point2d center(box.x + box.width / 2.0, box.y + box.height / 2.0);
    const cv::Point3d p = testRig().left().backProject(center, depth);
    Detection3D d;
    d.box = box;
    d.position =
        cv::Point3f(static_cast<float>(p.x), static_cast<float>(p.y), static_cast<float>(p.z));
    d.depth = static_cast<float>(depth);
    d.class_id = class_id;
    d.score = score;
    d.valid = true;
    return d;
}

Detection3D makeDetectionWithAppearance(cv::Rect2f box, double depth, cv::Scalar bgr,
                                        int class_id = 0, float score = 0.9f) {
    Detection3D d = makeDetection(box, depth, class_id, score);
    const cv::Mat solid_color_crop(40, 40, CV_8UC3, bgr);
    d.appearance = computeAppearanceDescriptor(solid_color_crop, cv::Rect2f(0, 0, 40, 40));
    return d;
}

TrackerParams basicParams(AssociationMethod method) {
    TrackerParams p;
    p.dt = 0.1;
    p.process_noise = 1.0;
    p.measurement_noise = 0.05;
    p.max_age = 5;
    p.min_hits = 3;
    p.association = method;
    p.gating_chi2 = 7.815;  // chi-square(3 dof, 95%)
    p.iou_gate = 0.3;
    p.use_hungarian = true;
    return p;
}

}  // namespace

TEST(Tracker, BirthIsTentativeThenConfirmsAfterMinHits) {
    Tracker tracker(basicParams(AssociationMethod::kMahalanobis3D));
    const Detection3D d = makeDetection(cv::Rect2f(300, 200, 40, 40), 3.0);

    const TrackerUpdateResult r0 = tracker.update({d});
    ASSERT_EQ(tracker.tracks().size(), 1u);
    EXPECT_FALSE(tracker.tracks()[0].confirmed());  // min_hits = 3, this is hit 1

    tracker.update({d});
    EXPECT_FALSE(tracker.tracks()[0].confirmed());  // hit 2

    const TrackerUpdateResult r2 = tracker.update({d});
    EXPECT_TRUE(tracker.tracks()[0].confirmed());                   // hit 3 -> confirmed
    EXPECT_EQ(r2.detection_track_id[0], r0.detection_track_id[0]);  // stable id throughout
}

TEST(Tracker, CoastsOnMissThenDiesAfterMaxAge) {
    TrackerParams p = basicParams(AssociationMethod::kMahalanobis3D);
    p.max_age = 2;
    Tracker tracker(p);
    const Detection3D d = makeDetection(cv::Rect2f(300, 200, 40, 40), 3.0);

    tracker.update({d});
    ASSERT_EQ(tracker.tracks().size(), 1u);

    tracker.update({});  // miss 1
    ASSERT_EQ(tracker.tracks().size(), 1u);
    EXPECT_EQ(tracker.tracks()[0].timeSinceUpdate(), 1);

    tracker.update({});  // miss 2 (== max_age, still kept)
    ASSERT_EQ(tracker.tracks().size(), 1u);
    EXPECT_EQ(tracker.tracks()[0].timeSinceUpdate(), 2);

    tracker.update({});  // miss 3 (> max_age -> pruned)
    EXPECT_TRUE(tracker.tracks().empty());
}

TEST(Tracker, InvalidDetectionsAreIgnored) {
    Tracker tracker(basicParams(AssociationMethod::kMahalanobis3D));
    Detection3D invalid = makeDetection(cv::Rect2f(300, 200, 40, 40), 3.0);
    invalid.valid = false;

    const TrackerUpdateResult r = tracker.update({invalid});
    EXPECT_TRUE(tracker.tracks().empty());
    ASSERT_EQ(r.detection_track_id.size(), 1u);
    EXPECT_EQ(r.detection_track_id[0], -1);
}

TEST(Tracker, DifferentClassNeverMatchesSpawnsSeparateTrack) {
    Tracker tracker(basicParams(AssociationMethod::kMahalanobis3D));
    const cv::Rect2f box(300, 200, 40, 40);
    const TrackerUpdateResult r0 = tracker.update({makeDetection(box, 3.0, /*class_id=*/0)});
    ASSERT_EQ(tracker.tracks().size(), 1u);

    // Same box, same depth, different class -- must NOT correct the existing
    // track; it should be missed this frame and a second track born instead.
    const TrackerUpdateResult r1 = tracker.update({makeDetection(box, 3.0, /*class_id=*/5)});
    ASSERT_EQ(tracker.tracks().size(), 2u);
    EXPECT_NE(r1.detection_track_id[0], r0.detection_track_id[0]);

    bool found_missed_original = false;
    for (const Track& t : tracker.tracks()) {
        if (t.id() == r0.detection_track_id[0]) {
            EXPECT_EQ(t.timeSinceUpdate(), 1);
            found_missed_original = true;
        }
    }
    EXPECT_TRUE(found_missed_original);
}

// The centrepiece scenario: a near object (2 m) and a far object (8 m) whose 2D
// boxes end up overlapping on screen. A single new detection sits exactly on
// the far track's last box (IoU = 1) yet carries the near object's true depth.
//
//   kIou2D          matches it to the FAR track  (wrong -- pure box overlap)
//   kMahalanobis3D  matches it to the NEAR track (right -- depth disambiguates)
//
// All numbers below are exact (integer pixel boxes, exact depths), chosen with
// a wide safety margin from both gates (IoU cost 0.72 vs a 0.70 gate; squared
// Mahalanobis ~14000 for the wrong pairing vs a 7.815 gate) -- see the
// worked arithmetic in the PR/commit description.
TEST(Tracker, DepthSeparatesOccludingBoxesIou2DGetsItWrong) {
    const cv::Rect2f near_box(280, 200, 40, 40);  // settles near track "A"
    const cv::Rect2f far_box(300, 205, 40, 40);   // settles far track "B" -- overlaps near_box
    const double near_depth = 2.0;
    const double far_depth = 8.0;

    for (AssociationMethod method :
         {AssociationMethod::kMahalanobis3D, AssociationMethod::kIou2D}) {
        TrackerParams p = basicParams(method);
        p.min_hits = 1;
        Tracker tracker(p);

        // Establish two separate, confirmed tracks with unmoving detections so
        // each settles near its own back-projected 3D position with ~zero
        // velocity.
        int id_near = -1;
        int id_far = -1;
        for (int f = 0; f < 3; ++f) {
            const TrackerUpdateResult r = tracker.update(
                {makeDetection(near_box, near_depth), makeDetection(far_box, far_depth)});
            id_near = r.detection_track_id[0];
            id_far = r.detection_track_id[1];
        }
        ASSERT_EQ(tracker.tracks().size(), 2u);
        ASSERT_NE(id_near, id_far);

        // The disambiguating detection: exactly the far track's box (IoU = 1
        // against it), but the near track's true depth.
        const Detection3D ambiguous = makeDetection(far_box, near_depth);
        const TrackerUpdateResult r = tracker.update({ambiguous});

        if (method == AssociationMethod::kMahalanobis3D) {
            EXPECT_EQ(r.detection_track_id[0], id_near)
                << "3D association should follow the matching depth, not screen overlap";
        } else {
            EXPECT_EQ(r.detection_track_id[0], id_far)
                << "2D IoU association has no depth cue and follows screen overlap instead";
        }
    }
}

// Phase 5: fusing in appearance recovers an identity that Mahalanobis alone
// gets wrong. Two tracks settle at close depths with the SAME box throughout
// (so IoU never discriminates between them) -- one red at 2.00 m, one blue at
// 2.10 m. The disambiguating detection is red, but sits geometrically closer
// to the *blue* track's settled depth (2.07 m: 0.07 m from red's 2.00 m vs
// only 0.03 m from blue's 2.10 m -- ~5.4x cheaper in squared Mahalanobis
// terms, so plain kMahalanobis3D confidently picks the wrong, blue track).
// kFusedAppearance's default weights (0.5 geometry / 0.2 IoU / 0.3 appearance)
// are enough for the appearance term (distance 0 to red, 1 to blue) to
// overturn that and pick the right, red one.
TEST(Tracker, FusedAppearanceRecoversIdentityMahalanobisAloneGetsWrong) {
    const cv::Rect2f box(280, 200, 40, 40);
    const cv::Scalar red(0, 0, 255), blue(255, 0, 0);

    TrackerParams p = basicParams(AssociationMethod::kFusedAppearance);
    p.min_hits = 1;
    Tracker tracker(p);

    int id_red = -1;
    int id_blue = -1;
    for (int f = 0; f < 3; ++f) {
        const TrackerUpdateResult r =
            tracker.update({makeDetectionWithAppearance(box, 2.00, red),
                            makeDetectionWithAppearance(box, 2.10, blue)});
        id_red = r.detection_track_id[0];
        id_blue = r.detection_track_id[1];
    }
    ASSERT_EQ(tracker.tracks().size(), 2u);
    ASSERT_NE(id_red, id_blue);

    const Detection3D ambiguous = makeDetectionWithAppearance(box, 2.07, red);
    const TrackerUpdateResult r = tracker.update({ambiguous});
    EXPECT_EQ(r.detection_track_id[0], id_red)
        << "appearance should override a small Mahalanobis-distance disadvantage";
}

// Same setup with no appearance descriptors at all (Detection3D's default
// empty Mat): the neutral 0.5 fallback contributes equally to every
// candidate, so it must not change the ranking -- kFusedAppearance should
// fall back to deciding on geometry alone, same outcome as kMahalanobis3D.
TEST(Tracker, FusedAppearanceFallsBackToGeometryWithNoDescriptor) {
    const cv::Rect2f box(280, 200, 40, 40);
    TrackerParams p = basicParams(AssociationMethod::kFusedAppearance);
    p.min_hits = 1;
    Tracker tracker(p);

    int id_near = -1;
    int id_far = -1;
    for (int f = 0; f < 3; ++f) {
        const TrackerUpdateResult r =
            tracker.update({makeDetection(box, 2.00), makeDetection(box, 2.10)});
        id_near = r.detection_track_id[0];
        id_far = r.detection_track_id[1];
    }
    ASSERT_NE(id_near, id_far);

    const TrackerUpdateResult r = tracker.update({makeDetection(box, 2.02)});
    EXPECT_EQ(r.detection_track_id[0], id_near);
}

TEST(Tracker, GreedyAndHungarianAgreeOnAnUnambiguousScene) {
    TrackerParams h = basicParams(AssociationMethod::kMahalanobis3D);
    TrackerParams g = h;
    g.use_hungarian = false;
    Tracker hungarian(h);
    Tracker greedy(g);

    const std::vector<Detection3D> dets{
        makeDetection(cv::Rect2f(100, 100, 30, 30), 2.0),
        makeDetection(cv::Rect2f(500, 350, 30, 30), 10.0),
    };
    for (int f = 0; f < 3; ++f) {
        const TrackerUpdateResult rh = hungarian.update(dets);
        const TrackerUpdateResult rg = greedy.update(dets);
        EXPECT_NE(rh.detection_track_id[0], rh.detection_track_id[1]);
        EXPECT_NE(rg.detection_track_id[0], rg.detection_track_id[1]);
    }
    EXPECT_EQ(hungarian.tracks().size(), 2u);
    EXPECT_EQ(greedy.tracks().size(), 2u);
}

TEST(Tracker, ResetClearsTracksAndRestartsIds) {
    Tracker tracker(basicParams(AssociationMethod::kMahalanobis3D));
    tracker.update({makeDetection(cv::Rect2f(300, 200, 40, 40), 3.0)});
    ASSERT_FALSE(tracker.tracks().empty());

    tracker.reset();
    EXPECT_TRUE(tracker.tracks().empty());

    const TrackerUpdateResult r =
        tracker.update({makeDetection(cv::Rect2f(300, 200, 40, 40), 3.0)});
    EXPECT_EQ(r.detection_track_id[0], 1);  // id sequence restarted
}

TEST(TrackerParams, FromConfigMapsAssociationString) {
    TrackingParams tp;
    tp.association = "iou2d";
    tp.use_hungarian = false;
    const TrackerParams p = TrackerParams::fromConfig(tp);
    EXPECT_EQ(p.association, AssociationMethod::kIou2D);
    EXPECT_FALSE(p.use_hungarian);

    tp.association = "mahalanobis3d";
    EXPECT_EQ(TrackerParams::fromConfig(tp).association, AssociationMethod::kMahalanobis3D);

    tp.association = "unrecognised-defaults-to-3d";
    EXPECT_EQ(TrackerParams::fromConfig(tp).association, AssociationMethod::kMahalanobis3D);

    tp.association = "fused";
    tp.fused_weight_3d = 0.4;
    tp.fused_weight_iou = 0.1;
    tp.fused_weight_appearance = 0.5;
    const TrackerParams fused = TrackerParams::fromConfig(tp);
    EXPECT_EQ(fused.association, AssociationMethod::kFusedAppearance);
    EXPECT_DOUBLE_EQ(fused.fused_weight_3d, 0.4);
    EXPECT_DOUBLE_EQ(fused.fused_weight_iou, 0.1);
    EXPECT_DOUBLE_EQ(fused.fused_weight_appearance, 0.5);
}

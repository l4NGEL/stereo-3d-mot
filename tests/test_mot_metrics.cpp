#include <vector>

#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include "s3m/tracking/mot_metrics.hpp"

using namespace s3m;

namespace {

MotObject obj(int id, float x, float y = 0.0f) {
    MotObject o;
    o.id = id;
    o.position = cv::Point3f(x, y, 0.0f);
    return o;
}

}  // namespace

// Every expected value below was produced by py-motmetrics (MOTAccumulator +
// mh.compute) on the identical frame sequence, with max_dist matching the
// gate used here -- these are regression pins against the reference
// implementation, not independently re-derived numbers.

TEST(MotMetrics, PerfectTrackingIsMotaOne) {
    MotAccumulator acc(1.5);
    acc.update({obj(1, 0), obj(2, 10)}, {obj(101, 0), obj(102, 10)});
    acc.update({obj(1, 1), obj(2, 11)}, {obj(101, 1), obj(102, 11)});
    acc.update({obj(1, 2), obj(2, 12)}, {obj(101, 2), obj(102, 12)});

    const MotSummary s = acc.summary();
    EXPECT_NEAR(s.mota, 1.0, 1e-9);
    EXPECT_NEAR(s.motp, 0.0, 1e-9);
    EXPECT_NEAR(s.idf1, 1.0, 1e-9);
    EXPECT_EQ(s.idtp, 6);
    EXPECT_EQ(s.num_matches, 6);
    EXPECT_EQ(s.num_switches, 0);
    EXPECT_EQ(s.num_fragmentations, 0);
    EXPECT_EQ(s.num_objects, 6);
}

TEST(MotMetrics, MissThenResumeSameIdIsOneFragmentationNoSwitch) {
    MotAccumulator acc(1.5);
    acc.update({obj(1, 0)}, {obj(101, 0)});
    acc.update({obj(1, 1)}, {});             // detector missed this frame
    acc.update({obj(1, 2)}, {obj(101, 2)});  // resumes under the SAME hyp id

    const MotSummary s = acc.summary();
    EXPECT_NEAR(s.mota, 2.0 / 3.0, 1e-9);
    EXPECT_NEAR(s.idf1, 0.8, 1e-9);
    EXPECT_EQ(s.idtp, 2);
    EXPECT_EQ(s.idfn, 1);
    EXPECT_EQ(s.num_matches, 2);
    EXPECT_EQ(s.num_switches, 0);
    EXPECT_EQ(s.num_fragmentations, 1);  // tracked -> not tracked -> tracked
    EXPECT_EQ(s.num_misses, 1);
}

TEST(MotMetrics, MissThenResumeDifferentIdIsASwitch) {
    MotAccumulator acc(1.5);
    acc.update({obj(1, 0)}, {obj(101, 0)});
    acc.update({obj(1, 1)}, {});
    acc.update({obj(1, 2)}, {obj(202, 2)});  // resumes under a DIFFERENT hyp id

    const MotSummary s = acc.summary();
    EXPECT_NEAR(s.mota, 1.0 / 3.0, 1e-9);
    EXPECT_NEAR(s.idf1, 0.4, 1e-9);
    EXPECT_EQ(s.idtp, 1);
    EXPECT_EQ(s.idfp, 1);
    EXPECT_EQ(s.idfn, 2);
    EXPECT_EQ(s.num_matches, 1);
    EXPECT_EQ(s.num_switches, 1);
    EXPECT_EQ(s.num_fragmentations, 1);
}

TEST(MotMetrics, ExtraHypothesisIsAFalsePositive) {
    MotAccumulator acc(1.5);
    acc.update({obj(1, 0)}, {obj(101, 0), obj(999, 50)});

    const MotSummary s = acc.summary();
    EXPECT_NEAR(s.mota, 0.0, 1e-9);
    EXPECT_NEAR(s.idf1, 2.0 / 3.0, 1e-9);
    EXPECT_EQ(s.num_matches, 1);
    EXPECT_EQ(s.num_false_positives, 1);
    EXPECT_NEAR(s.precision, 0.5, 1e-9);
    EXPECT_NEAR(s.recall, 1.0, 1e-9);
}

TEST(MotMetrics, ContinuityPreferenceSurvivesACrossing) {
    // Two objects swap relative screen position; in isolation the nearest-hyp
    // choice at the crossing frame would swap ids, but CLEAR-MOT's "keep the
    // previous correspondence if it's still valid" rule (step 1) must win.
    MotAccumulator acc(1.5);
    acc.update({obj(1, -5), obj(2, 5)}, {obj(101, -5), obj(102, 5)});
    acc.update({obj(1, -1), obj(2, 1)}, {obj(101, -1), obj(102, 1)});
    acc.update({obj(1, 1), obj(2, -1)}, {obj(101, 0.9f), obj(102, -0.9f)});  // crossed
    acc.update({obj(1, 5), obj(2, -5)}, {obj(101, 5), obj(102, -5)});

    const MotSummary s = acc.summary();
    EXPECT_NEAR(s.mota, 1.0, 1e-9);
    EXPECT_EQ(s.num_switches, 0);
    EXPECT_EQ(s.num_fragmentations, 0);
    EXPECT_EQ(s.num_matches, 8);
}

TEST(MotMetrics, OutOfGateBecomesMissPlusFalsePositive) {
    MotAccumulator acc(1.5);
    acc.update({obj(1, 0, 0)}, {obj(101, 100, 100)});  // far outside the gate

    const MotSummary s = acc.summary();
    EXPECT_NEAR(s.mota, -1.0, 1e-9);
    EXPECT_EQ(s.num_matches, 0);
    EXPECT_EQ(s.num_misses, 1);
    EXPECT_EQ(s.num_false_positives, 1);
    // No matched pairs: motp is 0 here by this class's convention (not NaN).
    EXPECT_NEAR(s.motp, 0.0, 1e-9);
}

TEST(MotMetrics, UnrelatedBirthAfterADeathIsNotASwitch) {
    MotAccumulator acc(1.5);
    acc.update({obj(1, 0)}, {obj(101, 0)});
    acc.update({}, {});
    acc.update({obj(2, 5, 5)}, {obj(202, 5, 5)});

    const MotSummary s = acc.summary();
    EXPECT_NEAR(s.mota, 1.0, 1e-9);
    EXPECT_EQ(s.num_switches, 0);
    EXPECT_EQ(s.num_matches, 2);
}

TEST(MotMetrics, ResetClearsAccumulatedState) {
    MotAccumulator acc(1.5);
    acc.update({obj(1, 0)}, {obj(101, 0)});
    acc.reset();
    acc.update({obj(1, 0)}, {obj(202, 0)});  // a fresh id after reset must not read as a switch

    const MotSummary s = acc.summary();
    EXPECT_EQ(s.num_switches, 0);
    EXPECT_EQ(s.num_objects, 1);
}

TEST(MotSummary, ToStringIsNonEmpty) {
    MotAccumulator acc(1.0);
    acc.update({obj(1, 0)}, {obj(101, 0)});
    EXPECT_FALSE(acc.summary().toString().empty());
}

#include <stdexcept>

#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include "s3m/tracking/kalman_filter.hpp"
#include "s3m/tracking/track.hpp"

using namespace s3m;

TEST(KalmanFilter, ConstantVelocityRecoversMotion) {
    const double dt = 0.1;
    KalmanFilter kf = makeConstantVelocity3D(dt, 1.0, 0.05);
    kf.init(KalmanFilter::Vec::Zero(6), KalmanFilter::Mat::Identity(6, 6) * 10.0);

    const double vx = 2.0;
    const double vy = -1.0;
    const double vz = 0.5;
    cv::RNG rng(123);
    for (int k = 1; k <= 120; ++k) {
        const double t = k * dt;
        KalmanFilter::Vec z(3);
        z << vx * t + rng.gaussian(0.05), vy * t + rng.gaussian(0.05), vz * t + rng.gaussian(0.05);
        kf.predict();
        kf.update(z);
    }

    const KalmanFilter::Vec& x = kf.state();
    EXPECT_NEAR(x(3), vx, 0.2);
    EXPECT_NEAR(x(4), vy, 0.2);
    EXPECT_NEAR(x(5), vz, 0.2);
    EXPECT_NEAR(x(0), vx * 12.0, 0.3);
}

TEST(KalmanFilter, GatingDistanceGrowsWithOffset) {
    KalmanFilter kf = makeConstantVelocity3D(0.1, 1.0, 0.1);
    kf.init(KalmanFilter::Vec::Zero(6), KalmanFilter::Mat::Identity(6, 6) * 0.01);
    kf.predict();

    KalmanFilter::Vec z_near(3);
    z_near << 0.01, 0.0, 0.0;
    KalmanFilter::Vec z_far(3);
    z_far << 5.0, 5.0, 5.0;

    EXPECT_LT(kf.gatingDistanceSq(z_near), kf.gatingDistanceSq(z_far));
    EXPECT_GE(kf.gatingDistanceSq(z_near), 0.0);
}

TEST(KalmanFilter, RejectsNonPositiveDimensions) {
    EXPECT_THROW(KalmanFilter(0, 3), std::invalid_argument);
    EXPECT_THROW(KalmanFilter(6, 0), std::invalid_argument);
}

TEST(Track, PredictsCorrectsAndConfirms) {
    Track track(1, cv::Point3f(1.0f, 2.0f, 5.0f), 0.1, 1.0, 0.05);
    track.setMinHits(3);
    EXPECT_EQ(track.id(), 1);
    EXPECT_FALSE(track.confirmed());

    for (int i = 0; i < 3; ++i) {
        track.predict();
        track.correct(cv::Point3f(1.0f, 2.0f, 5.0f));
    }

    EXPECT_TRUE(track.confirmed());
    EXPECT_EQ(track.timeSinceUpdate(), 0);
    EXPECT_GE(track.hits(), 3);

    const cv::Point3f p = track.position();
    EXPECT_NEAR(p.x, 1.0f, 0.3f);
    EXPECT_NEAR(p.y, 2.0f, 0.3f);
    EXPECT_NEAR(p.z, 5.0f, 0.3f);
}

TEST(Track, MarkMissedIncrementsStaleness) {
    Track track(2, cv::Point3f(0.0f, 0.0f, 3.0f), 0.1, 1.0, 0.05);
    track.predict();
    track.markMissed();
    track.predict();
    track.markMissed();
    EXPECT_EQ(track.timeSinceUpdate(), 2);
    EXPECT_GE(track.age(), 2);
}

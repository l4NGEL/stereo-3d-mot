#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include <opencv2/imgcodecs.hpp>

#include "s3m/io/kitti_odometry_loader.hpp"

using namespace s3m;
namespace fs = std::filesystem;

namespace {

// A tiny, hand-built KITTI odometry sequence "04" with three frames.
// Calibration uses P0/P1 (grayscale), not P2/P3 -- the point of this loader.
// Same fx=fy=700, cx_left=600, cx_right=590, cy=180, baseline=0.54m
// convention as the tracking-benchmark fixture, so the two loaders' numbers
// are directly comparable.
class KittiOdometryFixture : public ::testing::Test {
 protected:
    void SetUp() override {
        root_ = fs::temp_directory_path() / "s3m_kitti_odometry_fixture";
        fs::remove_all(root_);
        fs::create_directories(root_ / "sequences" / "04" / "image_0");
        fs::create_directories(root_ / "sequences" / "04" / "image_1");
        fs::create_directories(root_ / "poses");

        std::ofstream calib(root_ / "sequences" / "04" / "calib.txt");
        calib << "P0: 700.0 0.0 600.0 0.0 0.0 700.0 180.0 0.0 0.0 0.0 1.0 0.0\n"
              << "P1: 700.0 0.0 600.0 -378.0 0.0 700.0 180.0 0.0 0.0 0.0 1.0 0.0\n"
              << "P2: 700.0 0.0 600.0 0.0 0.0 700.0 180.0 0.0 0.0 0.0 1.0 0.0\n"
              << "P3: 700.0 0.0 590.0 -378.0 0.0 700.0 180.0 0.0 0.0 0.0 1.0 0.0\n";
        calib.close();

        std::ofstream times(root_ / "sequences" / "04" / "times.txt");
        times << "0.000000e+00\n1.041284e-01\n2.082536e-01\n";
        times.close();

        // Frame 0: identity. Frame 1: camera moved +1m along world X.
        // Frame 2: +2m along X, no rotation (kept simple -- rotation
        // composition is covered by the trajectory_eval tests instead).
        std::ofstream poses(root_ / "poses" / "04.txt");
        poses << "1 0 0 0 0 1 0 0 0 0 1 0\n"
              << "1 0 0 1 0 1 0 0 0 0 1 0\n"
              << "1 0 0 2 0 1 0 0 0 0 1 0\n";
        poses.close();

        const cv::Mat left(64, 96, CV_8UC3, cv::Scalar(60, 90, 120));
        const cv::Mat right(64, 96, CV_8UC3, cv::Scalar(60, 90, 120));
        for (const char* name : {"000000.png", "000001.png", "000002.png"}) {
            cv::imwrite((root_ / "sequences" / "04" / "image_0" / name).string(), left);
            cv::imwrite((root_ / "sequences" / "04" / "image_1" / name).string(), right);
        }
    }

    void TearDown() override { fs::remove_all(root_); }

    fs::path root_;
};

}  // namespace

TEST_F(KittiOdometryFixture, ReadKittiOdometryCalibUsesP0P1NotP2P3) {
    const StereoRig rig =
        readKittiOdometryCalib((root_ / "sequences" / "04" / "calib.txt").string(), cv::Size(96, 64));
    EXPECT_NEAR(rig.left().fx(), 700.0, 1e-6);
    EXPECT_NEAR(rig.left().cx(), 600.0, 1e-6);
    EXPECT_NEAR(rig.right().cx(), 600.0, 1e-6);  // P1's cx, same as P0 here -- distinct from P3's 590
    EXPECT_NEAR(rig.baseline(), 0.54, 1e-6);
}

TEST(ReadKittiPoses, ParsesTranslationAndRotationColumns) {
    const fs::path root = fs::temp_directory_path() / "s3m_kitti_poses_only";
    fs::remove_all(root);
    fs::create_directories(root);
    std::ofstream f(root / "p.txt");
    // Row-major [R|t]: R = 90-degree rotation about Y (x'=z, z'=-x), t = (1,2,3).
    f << "0 0 1 1   0 1 0 2   -1 0 0 3\n";
    f.close();

    const std::vector<Pose> poses = readKittiPoses((root / "p.txt").string());
    ASSERT_EQ(poses.size(), 1u);
    EXPECT_DOUBLE_EQ(poses[0].position.x, 1.0);
    EXPECT_DOUBLE_EQ(poses[0].position.y, 2.0);
    EXPECT_DOUBLE_EQ(poses[0].position.z, 3.0);
    EXPECT_NEAR(poses[0].rotation(0, 2), 1.0, 1e-9);
    EXPECT_NEAR(poses[0].rotation(2, 0), -1.0, 1e-9);
    fs::remove_all(root);
}

TEST_F(KittiOdometryFixture, SourceEnumeratesFramesAndLoadsGroundTruth) {
    KittiOdometrySource src(root_.string(), "04");
    EXPECT_EQ(src.size(), 3);
    EXPECT_TRUE(src.hasGroundTruth());
    EXPECT_NEAR(src.rig().baseline(), 0.54, 1e-6);

    ASSERT_EQ(src.groundTruthPoses().size(), 3u);
    EXPECT_DOUBLE_EQ(src.groundTruthPoses()[1].position.x, 1.0);
    EXPECT_DOUBLE_EQ(src.groundTruthPoses()[2].position.x, 2.0);

    const std::optional<StereoFrame> f0 = src.next();
    ASSERT_TRUE(f0.has_value());
    EXPECT_EQ(f0->index, 0);
    EXPECT_EQ(f0->left.size(), cv::Size(96, 64));

    src.next();
    src.next();
    EXPECT_FALSE(src.next().has_value());  // exhausted after 3 frames

    src.reset();
    EXPECT_TRUE(src.next().has_value());
}

TEST_F(KittiOdometryFixture, MismatchedPoseCountThrows) {
    std::ofstream poses(root_ / "poses" / "04.txt");  // overwrite: only 1 pose for 3 frames
    poses << "1 0 0 0 0 1 0 0 0 0 1 0\n";
    poses.close();
    EXPECT_THROW(KittiOdometrySource(root_.string(), "04"), std::runtime_error);
}

TEST(KittiOdometrySource, MissingSequenceThrows) {
    const fs::path root = fs::temp_directory_path() / "s3m_kitti_odometry_missing";
    fs::remove_all(root);
    fs::create_directories(root);
    EXPECT_THROW(KittiOdometrySource(root.string(), "04"), std::runtime_error);
    fs::remove_all(root);
}

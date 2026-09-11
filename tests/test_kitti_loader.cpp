#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include <opencv2/imgcodecs.hpp>

#include "s3m/io/kitti_loader.hpp"

using namespace s3m;
namespace fs = std::filesystem;

namespace {

// A tiny, hand-built KITTI tracking sequence "0000" with two frames.
// Calibration is self-consistent with the documented rectified-P convention
// (P = K * [I | t], t = (-baseline, 0, 0)): fx=fy=700, cy=180 shared;
// cx_left=600, cx_right=590 (doffs=-10, deliberately non-zero to exercise
// that path); baseline = 378/700 = 0.54 m (KITTI's real rig baseline).
class KittiFixture : public ::testing::Test {
 protected:
    void SetUp() override {
        root_ = fs::temp_directory_path() / "s3m_kitti_fixture";
        fs::remove_all(root_);
        fs::create_directories(root_ / "calib");
        fs::create_directories(root_ / "image_02" / "0000");
        fs::create_directories(root_ / "image_03" / "0000");
        fs::create_directories(root_ / "label_02");

        std::ofstream calib(root_ / "calib" / "0000.txt");
        calib << "P0: 700.0 0.0 600.0 0.0 0.0 700.0 180.0 0.0 0.0 0.0 1.0 0.0\n"
              << "P1: 700.0 0.0 600.0 -378.0 0.0 700.0 180.0 0.0 0.0 0.0 1.0 0.0\n"
              << "P2: 700.0 0.0 600.0 0.0 0.0 700.0 180.0 0.0 0.0 0.0 1.0 0.0\n"
              << "P3: 700.0 0.0 590.0 -378.0 0.0 700.0 180.0 0.0 0.0 0.0 1.0 0.0\n"
              << "R_rect: 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0\n"
              << "Tr_velo_cam: 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0\n"
              << "Tr_imu_velo: 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0\n";
        calib.close();

        std::ofstream labels(root_ / "label_02" / "0000.txt");
        labels << "0 1 Car 0.0 0 -1.5 100.0 100.0 200.0 200.0 1.5 1.6 3.5 5.0 1.8 20.0 0.0\n"
               << "0 -1 DontCare -1 -1 -10 50.0 50.0 80.0 90.0 -1 -1 -1 -1000 -1000 -1000 -10\n"
               << "1 1 Car 0.0 0 -1.5 105.0 100.0 205.0 200.0 1.5 1.6 3.5 5.2 1.8 20.0 0.0\n"
               << "1 2 Pedestrian 0.0 0 0.0 300.0 150.0 330.0 220.0 1.8 0.6 0.6 -2.0 1.7 8.0 0.0\n"
               << "this line is garbage and should be skipped\n";
        labels.close();

        const cv::Mat left(64, 96, CV_8UC3, cv::Scalar(60, 90, 120));
        const cv::Mat right(64, 96, CV_8UC3, cv::Scalar(60, 90, 120));
        for (const char* name : {"000000.png", "000001.png"}) {
            cv::imwrite((root_ / "image_02" / "0000" / name).string(), left);
            cv::imwrite((root_ / "image_03" / "0000" / name).string(), right);
        }
    }

    void TearDown() override { fs::remove_all(root_); }

    fs::path root_;
};

}  // namespace

TEST_F(KittiFixture, ReadLabelsParsesAllValidLinesSkipsGarbage) {
    const std::vector<KittiObject> objs = readKittiLabels((root_ / "label_02" / "0000.txt").string());
    ASSERT_EQ(objs.size(), 4u);  // 5 lines in the file, 1 garbage line skipped

    EXPECT_EQ(objs[0].frame, 0);
    EXPECT_EQ(objs[0].track_id, 1);
    EXPECT_EQ(objs[0].type, "Car");
    EXPECT_FALSE(objs[0].isDontCare());
    EXPECT_FLOAT_EQ(objs[0].bbox.x, 100.0f);
    EXPECT_FLOAT_EQ(objs[0].bbox.width, 100.0f);   // 200 - 100
    EXPECT_DOUBLE_EQ(objs[0].location.z, 20.0);
    EXPECT_LT(objs[0].score, 0.0);                  // GT file: no trailing score

    EXPECT_TRUE(objs[1].isDontCare());
    EXPECT_EQ(objs[1].track_id, -1);

    EXPECT_EQ(objs[3].type, "Pedestrian");
    EXPECT_EQ(objs[3].track_id, 2);
}

TEST_F(KittiFixture, ReadLabelsThrowsOnMissingFile) {
    EXPECT_THROW(readKittiLabels("/no/such/label_file.txt"), std::runtime_error);
}

TEST_F(KittiFixture, CentroidLiftsAboveBottomCenterByHalfHeight) {
    const KittiObject o = readKittiLabels((root_ / "label_02" / "0000.txt").string()).front();
    const cv::Point3d c = o.centroid();
    EXPECT_DOUBLE_EQ(c.x, o.location.x);
    EXPECT_DOUBLE_EQ(c.y, o.location.y - o.height / 2.0);
    EXPECT_DOUBLE_EQ(c.z, o.location.z);
}

TEST_F(KittiFixture, ReadCalibRecoversFocalLengthBaselineAndDoffs) {
    const StereoRig rig = readKittiCalib((root_ / "calib" / "0000.txt").string(), cv::Size(96, 64));
    EXPECT_NEAR(rig.left().fx(), 700.0, 1e-6);
    EXPECT_NEAR(rig.left().fy(), 700.0, 1e-6);
    EXPECT_NEAR(rig.left().cx(), 600.0, 1e-6);
    EXPECT_NEAR(rig.right().cx(), 590.0, 1e-6);
    EXPECT_NEAR(rig.baseline(), 0.54, 1e-6);   // (0 - (-378)) / 700
    EXPECT_NEAR(rig.doffs(), -10.0, 1e-6);     // cx_right - cx_left
}

TEST_F(KittiFixture, ReadCalibThrowsWhenP2OrP3Missing) {
    const fs::path bad = root_ / "calib" / "bad.txt";
    std::ofstream f(bad);
    f << "P0: 700.0 0.0 600.0 0.0 0.0 700.0 180.0 0.0 0.0 0.0 1.0 0.0\n";
    f.close();
    EXPECT_THROW(readKittiCalib(bad.string(), cv::Size(96, 64)), std::runtime_error);
}

TEST_F(KittiFixture, SourceEnumeratesFramesAndBuildsRig) {
    KittiTrackingSource src(root_.string(), "0000");
    EXPECT_EQ(src.size(), 2);
    EXPECT_TRUE(src.hasGroundTruth());
    EXPECT_NEAR(src.rig().baseline(), 0.54, 1e-6);

    const std::optional<StereoFrame> f0 = src.next();
    ASSERT_TRUE(f0.has_value());
    EXPECT_EQ(f0->index, 0);
    EXPECT_FALSE(f0->left.empty());
    EXPECT_FALSE(f0->right.empty());
    EXPECT_EQ(f0->left.size(), cv::Size(96, 64));

    const std::optional<StereoFrame> f1 = src.next();
    ASSERT_TRUE(f1.has_value());
    EXPECT_EQ(f1->index, 1);

    EXPECT_FALSE(src.next().has_value());  // exhausted

    src.reset();
    EXPECT_TRUE(src.next().has_value());
}

TEST_F(KittiFixture, ObjectsAtExcludesDontCareAndIndexesBySequencePosition) {
    KittiTrackingSource src(root_.string(), "0000");

    const std::vector<KittiObject> f0 = src.objectsAt(0);
    ASSERT_EQ(f0.size(), 1u);  // the DontCare on frame 0 is excluded
    EXPECT_EQ(f0[0].track_id, 1);

    const std::vector<KittiObject> f1 = src.objectsAt(1);
    EXPECT_EQ(f1.size(), 2u);

    EXPECT_TRUE(src.objectsAt(99).empty());
    EXPECT_TRUE(src.objectsAt(-1).empty());
}

TEST(KittiTrackingSource, MissingSequenceThrows) {
    const fs::path root = fs::temp_directory_path() / "s3m_kitti_missing";
    fs::remove_all(root);
    fs::create_directories(root);
    EXPECT_THROW(KittiTrackingSource(root.string(), "0000"), std::runtime_error);
    fs::remove_all(root);
}

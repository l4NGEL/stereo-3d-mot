#include <cmath>
#include <stdexcept>

#include <gtest/gtest.h>

#include "s3m/camera/camera_model.hpp"

using s3m::CameraModel;

TEST(CameraModel, ProjectBackProjectRoundTrip) {
    const CameraModel cam(600.0, 600.0, 320.0, 240.0, cv::Size(640, 480));
    const cv::Point3d point(0.5, -0.3, 4.0);
    const cv::Point2d pixel = cam.project(point);
    const cv::Point3d recovered = cam.backProject(pixel, point.z);
    EXPECT_NEAR(recovered.x, point.x, 1e-9);
    EXPECT_NEAR(recovered.y, point.y, 1e-9);
    EXPECT_NEAR(recovered.z, point.z, 1e-9);
}

TEST(CameraModel, PrincipalPointProjectsToImageCentre) {
    const CameraModel cam(500.0, 500.0, 320.0, 240.0, cv::Size(640, 480));
    const cv::Point2d pixel = cam.project(cv::Point3d(0.0, 0.0, 10.0));
    EXPECT_NEAR(pixel.x, 320.0, 1e-9);
    EXPECT_NEAR(pixel.y, 240.0, 1e-9);
}

TEST(CameraModel, RayIsUnitLengthAndReprojects) {
    const CameraModel cam(500.0, 520.0, 300.0, 260.0, cv::Size(600, 520));
    const cv::Point2d pixel(123.0, 456.0);
    const cv::Point3d dir = cam.ray(pixel);
    EXPECT_NEAR(std::sqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z), 1.0, 1e-12);

    const double s = 7.0 / dir.z;
    const cv::Point2d reprojected = cam.project(cv::Point3d(dir.x * s, dir.y * s, dir.z * s));
    EXPECT_NEAR(reprojected.x, pixel.x, 1e-7);
    EXPECT_NEAR(reprojected.y, pixel.y, 1e-7);
}

TEST(CameraModel, ProjectRejectsPointBehindCamera) {
    const CameraModel cam(500.0, 500.0, 320.0, 240.0, cv::Size(640, 480));
    EXPECT_THROW(cam.project(cv::Point3d(0.0, 0.0, -1.0)), std::invalid_argument);
}

TEST(CameraModel, ScaledIntrinsics) {
    const CameraModel cam(500.0, 500.0, 320.0, 240.0, cv::Size(640, 480));
    const CameraModel half = cam.scaled(0.5, 0.5);
    EXPECT_DOUBLE_EQ(half.fx(), 250.0);
    EXPECT_DOUBLE_EQ(half.cx(), 160.0);
    EXPECT_EQ(half.imageSize().width, 320);
    EXPECT_EQ(half.imageSize().height, 240);
}

TEST(CameraModel, FromMatrix) {
    const cv::Matx33d k(700.0, 0.0, 333.0, 0.0, 710.0, 244.0, 0.0, 0.0, 1.0);
    const CameraModel cam = CameraModel::fromMatrix(k, cv::Size(666, 500));
    EXPECT_DOUBLE_EQ(cam.fx(), 700.0);
    EXPECT_DOUBLE_EQ(cam.fy(), 710.0);
    EXPECT_DOUBLE_EQ(cam.cx(), 333.0);
    EXPECT_DOUBLE_EQ(cam.cy(), 244.0);
}

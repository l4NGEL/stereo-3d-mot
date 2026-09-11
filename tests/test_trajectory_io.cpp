#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "s3m/io/trajectory_io.hpp"

using namespace s3m;

namespace {

TrackState makeState(int id, float x, float y, float z, int class_id, bool confirmed) {
    TrackState s;
    s.id = id;
    s.position = cv::Point3f(x, y, z);
    s.velocity = cv::Point3f(0.1f, 0.0f, 0.0f);
    s.class_id = class_id;
    s.confirmed = confirmed;
    return s;
}

}  // namespace

TEST(TrajectoryRecorder, GroupsPointsByTrackIdInFrameOrder) {
    TrajectoryRecorder rec;
    rec.record(0, 0.0, {makeState(1, 0.0f, 0.0f, 2.0f, 0, false)});
    rec.record(1, 0.1, {makeState(1, 0.1f, 0.0f, 2.0f, 0, true), makeState(2, 5.0f, 0.0f, 9.0f, 2, false)});

    ASSERT_EQ(rec.trackCount(), 2u);
    const auto& traj = rec.trajectories();
    ASSERT_EQ(traj.at(1).size(), 2u);
    ASSERT_EQ(traj.at(2).size(), 1u);
    EXPECT_EQ(traj.at(1)[0].frame, 0);
    EXPECT_EQ(traj.at(1)[1].frame, 1);
    EXPECT_FLOAT_EQ(traj.at(1)[1].state.position.x, 0.1f);
}

TEST(TrajectoryIo, CsvRoundTrip) {
    TrajectoryRecorder rec;
    rec.record(0, 0.0, {makeState(7, 1.0f, 2.0f, 3.0f, 0, false)});
    rec.record(1, 0.1, {makeState(7, 1.1f, 2.0f, 3.0f, 0, true)});

    const std::string path = "s3m_test_trajectories.csv";
    writeTrajectoriesCsv(path, rec);

    std::ifstream file(path);
    ASSERT_TRUE(file.good());
    std::string header;
    std::getline(file, header);
    EXPECT_EQ(header, "track_id,frame,timestamp,x,y,z,vx,vy,vz,class_id,confirmed");

    std::string row1, row2;
    std::getline(file, row1);
    std::getline(file, row2);
    file.close();
    std::remove(path.c_str());

    ASSERT_FALSE(row1.empty());
    ASSERT_FALSE(row2.empty());
    EXPECT_NE(row1.find("7,0,0.000000,1.0000,2.0000,3.0000"), std::string::npos);
    EXPECT_NE(row2.find("7,1,0.100000,1.1000,2.0000,3.0000"), std::string::npos);
    EXPECT_EQ(row1.back(), '0');  // confirmed=0 (class_id,confirmed are the trailing fields)
    EXPECT_EQ(row2.back(), '1');  // confirmed=1
}

TEST(TrajectoryIo, PlyKeepsOnlyTracksWithAtLeastTwoPoints) {
    TrajectoryRecorder rec;
    rec.record(0, 0.0, {makeState(1, 0.0f, 0.0f, 2.0f, 0, true)});
    rec.record(1, 0.1, {makeState(1, 0.1f, 0.0f, 2.0f, 0, true)});
    rec.record(0, 0.0, {makeState(2, 5.0f, 0.0f, 9.0f, 0, true)});  // only ever seen once

    const std::string path = "s3m_test_trajectories.ply";
    writeTrajectoriesPly(path, rec);

    std::ifstream file(path);
    ASSERT_TRUE(file.good());
    int vertex_count = -1;
    int edge_count = -1;
    std::string line;
    while (std::getline(file, line)) {
        if (line.rfind("element vertex", 0) == 0) {
            std::istringstream iss(line);
            std::string a, b;
            iss >> a >> b >> vertex_count;
        } else if (line.rfind("element edge", 0) == 0) {
            std::istringstream iss(line);
            std::string a, b;
            iss >> a >> b >> edge_count;
        } else if (line == "end_header") {
            break;
        }
    }
    file.close();
    std::remove(path.c_str());

    ASSERT_GE(vertex_count, 0);
    ASSERT_GE(edge_count, 0);
    EXPECT_EQ(vertex_count, 2);  // only track 1's two points; track 2 excluded
    EXPECT_EQ(edge_count, 1);    // one segment connecting them
}

TEST(TrajectoryIo, EmptyRecorderWritesAValidEmptyFile) {
    const TrajectoryRecorder rec;
    const std::string csv = "s3m_test_empty.csv";
    const std::string ply = "s3m_test_empty.ply";
    writeTrajectoriesCsv(csv, rec);
    writeTrajectoriesPly(ply, rec);

    std::ifstream csv_file(csv);
    std::string header;
    std::getline(csv_file, header);
    EXPECT_EQ(header, "track_id,frame,timestamp,x,y,z,vx,vy,vz,class_id,confirmed");
    EXPECT_TRUE(csv_file.eof() || csv_file.peek() == std::ifstream::traits_type::eof());
    csv_file.close();

    std::ifstream ply_file(ply);
    EXPECT_TRUE(ply_file.good());
    ply_file.close();

    std::remove(csv.c_str());
    std::remove(ply.c_str());
}

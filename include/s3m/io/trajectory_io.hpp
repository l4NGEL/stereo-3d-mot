#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "s3m/core/types.hpp"

namespace s3m {

/// One track's state at one frame, as accumulated by TrajectoryRecorder.
struct TrajectoryPoint {
    std::int64_t frame = 0;
    double timestamp = 0.0;
    TrackState state;
};

/// Accumulates per-frame Tracker::update() output into per-track trajectories,
/// for export once a run finishes.
class TrajectoryRecorder {
 public:
    void record(std::int64_t frame, double timestamp, const std::vector<TrackState>& states);

    const std::map<int, std::vector<TrajectoryPoint>>& trajectories() const { return by_id_; }
    std::size_t trackCount() const { return by_id_.size(); }
    void clear() { by_id_.clear(); }

 private:
    std::map<int, std::vector<TrajectoryPoint>> by_id_;  ///< track id -> time-ordered points
};

/// Write every recorded point as one CSV row:
///   track_id,frame,timestamp,x,y,z,vx,vy,vz,class_id,confirmed
/// Throws std::runtime_error if the file cannot be opened.
void writeTrajectoriesCsv(const std::string& path, const TrajectoryRecorder& recorder);

/// Write every recorded track with >= 2 points as a PLY polyline: a `vertex`
/// element (positions, per-track colour) plus an `edge` element connecting
/// consecutive points of the same track, viewable alongside the point-cloud
/// exports (MeshLab / CloudCompare render `edge` elements as line segments).
/// Throws std::runtime_error if the file cannot be opened.
void writeTrajectoriesPly(const std::string& path, const TrajectoryRecorder& recorder);

}  // namespace s3m

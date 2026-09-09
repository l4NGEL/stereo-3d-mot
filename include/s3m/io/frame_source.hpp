#pragma once

#include <optional>

#include "s3m/camera/stereo_rig.hpp"
#include "s3m/core/types.hpp"

namespace s3m {

/// Abstract source of rectified stereo frames: dataset readers, the synthetic
/// scene, and (a later phase) a live camera all implement this.
class FrameSource {
 public:
    virtual ~FrameSource() = default;

    /// Calibration shared by every frame this source produces.
    virtual const StereoRig& rig() const = 0;

    /// Number of frames, or -1 when unbounded / unknown.
    virtual int size() const { return -1; }

    /// Next frame, or std::nullopt at end of stream.
    virtual std::optional<StereoFrame> next() = 0;

    /// Rewind to the first frame (no-op for live sources).
    virtual void reset() {}
};

}  // namespace s3m

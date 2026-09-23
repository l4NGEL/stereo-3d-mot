#!/usr/bin/env python3
"""Runtime smoke test for perception_node: publishes a few synthetic stereo
pairs on /stereo/left and /stereo/right, and reports whether
/perception/{detections,tracks,pointcloud} produced any messages in
response. Not a unit test (no assertions library, no CI wiring) -- a manual
verification tool, run once against the built node to confirm the whole
ROS2 graph actually moves data end to end, the thing a pure compile check
can't tell you. See docs/roadmap.md Phase 8 for how/when this was run.
"""
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from vision_msgs.msg import Detection3DArray


def make_image_msg(node, width=640, height=192, shift=0):
    # A vertically-striped pattern so ORB/SGBM have texture to match, with a
    # horizontal shift between "left" and "right" calls to fake a disparity.
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for x in range(0, width, 16):
        img[:, x:x + 8] = 200
    if shift:
        img = np.roll(img, shift, axis=1)
    msg = Image()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = "camera"
    msg.height, msg.width = height, width
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = width * 3
    msg.data = img.tobytes()
    return msg


class SmokeTest(Node):
    def __init__(self):
        super().__init__("s3m_smoke_test")
        self.left_pub = self.create_publisher(Image, "/stereo/left", 10)
        self.right_pub = self.create_publisher(Image, "/stereo/right", 10)
        self.counts = {"detections": 0, "tracks": 0, "pointcloud": 0}
        self.create_subscription(Detection3DArray, "/perception/detections",
                                 lambda m: self._bump("detections", len(m.detections)), 10)
        self.create_subscription(Detection3DArray, "/perception/tracks",
                                 lambda m: self._bump("tracks", len(m.detections)), 10)
        self.create_subscription(PointCloud2, "/perception/pointcloud",
                                 lambda m: self._bump("pointcloud", m.width), 10)

    def _bump(self, key, n):
        self.counts[key] += 1
        print(f"  received /{key}: message #{self.counts[key]} ({n} items)", flush=True)

    def publish_pair(self, i):
        self.left_pub.publish(make_image_msg(self, shift=0))
        self.right_pub.publish(make_image_msg(self, shift=6))
        print(f"published stereo pair {i}", flush=True)


def main():
    rclpy.init()
    node = SmokeTest()
    for i in range(5):
        node.publish_pair(i)
        rclpy.spin_once(node, timeout_sec=2.0)
        rclpy.spin_once(node, timeout_sec=1.0)
    ok = node.counts["pointcloud"] > 0
    print("\nresult:", node.counts)
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

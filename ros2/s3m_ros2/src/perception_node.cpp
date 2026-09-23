// Phase 8 (optional/bonus): a ROS2 node wrapping stereo-3d-mot's existing
// pipeline. Deliberately thin -- no perception logic lives here, only
// message <-> s3m-type conversion and wiring. Subscribes to a synchronized
// stereo pair, runs stereo depth -> detection -> 3D promotion -> tracking
// (exactly the same s3m::s3m library every other app in this repo uses),
// publishes detections, tracks, and a point cloud.
//
// Known, stated simplification: camera intrinsics/baseline come from ROS
// parameters, not a subscribed sensor_msgs/CameraInfo topic -- a real
// deployment would take calibration from CameraInfo so it can change at
// runtime; this node assumes a fixed, pre-calibrated rig, matching every
// other app in this project (they all take calibration from a file, not a
// live topic). Detection3D also has no 3D extent (position only, same as
// the rest of this project's tracker/metrics), so published bounding boxes
// use a fixed nominal size rather than a measured one.

#include <memory>
#include <string>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>

#include "s3m/camera/stereo_rig.hpp"
#include "s3m/core/config.hpp"
#include "s3m/depth/stereo_matcher.hpp"
#include "s3m/detection/detector.hpp"
#include "s3m/geometry/reprojection.hpp"
#include "s3m/tracking/tracker.hpp"
#if defined(S3M_WITH_ONNX)
#include "s3m/detection/onnx_detector.hpp"
#endif

using namespace s3m;
using sensor_msgs::msg::Image;
using ApproxSync =
    message_filters::sync_policies::ApproximateTime<Image, Image>;

namespace {

vision_msgs::msg::Detection3DArray toDetectionArray(const std::vector<Detection3D>& dets,
                                                     const std_msgs::msg::Header& header,
                                                     const std::vector<int>* track_ids = nullptr) {
    vision_msgs::msg::Detection3DArray msg;
    msg.header = header;
    msg.detections.reserve(dets.size());
    for (std::size_t i = 0; i < dets.size(); ++i) {
        const Detection3D& d = dets[i];
        vision_msgs::msg::Detection3D out;
        out.header = header;
        if (track_ids) {
            out.id = std::to_string((*track_ids)[i]);
        } else {
            out.id = std::to_string(i);
        }
        out.bbox.center.position.x = d.position.x;
        out.bbox.center.position.y = d.position.y;
        out.bbox.center.position.z = d.position.z;
        out.bbox.center.orientation.w = 1.0;
        // Detection3D carries no measured 3D extent (position-only, same
        // convention as the rest of this project) -- nominal placeholder
        // size, stated in the file header comment, not silently invented.
        out.bbox.size.x = 1.0;
        out.bbox.size.y = 1.0;
        out.bbox.size.z = 1.0;

        vision_msgs::msg::ObjectHypothesisWithPose hyp;
        hyp.hypothesis.class_id = std::to_string(d.class_id);
        hyp.hypothesis.score = static_cast<double>(d.score);
        hyp.pose.pose = out.bbox.center;
        out.results.push_back(hyp);

        msg.detections.push_back(std::move(out));
    }
    return msg;
}

sensor_msgs::msg::PointCloud2 toPointCloud2(const cv::Mat& points3d, const cv::Mat& valid_mask,
                                            const std_msgs::msg::Header& header) {
    sensor_msgs::msg::PointCloud2 msg;
    msg.header = header;
    msg.height = 1;

    int count = 0;
    for (int y = 0; y < valid_mask.rows; ++y) {
        const uchar* row = valid_mask.ptr<uchar>(y);
        for (int x = 0; x < valid_mask.cols; ++x) count += row[x] ? 1 : 0;
    }
    msg.width = static_cast<std::uint32_t>(count);
    msg.is_dense = true;
    msg.is_bigendian = false;

    sensor_msgs::PointCloud2Modifier modifier(msg);
    modifier.setPointCloud2FieldsByString(1, "xyz");
    modifier.resize(static_cast<std::size_t>(count));

    sensor_msgs::PointCloud2Iterator<float> it_x(msg, "x");
    sensor_msgs::PointCloud2Iterator<float> it_y(msg, "y");
    sensor_msgs::PointCloud2Iterator<float> it_z(msg, "z");
    for (int y = 0; y < valid_mask.rows; ++y) {
        const uchar* mrow = valid_mask.ptr<uchar>(y);
        const cv::Vec3f* prow = points3d.ptr<cv::Vec3f>(y);
        for (int x = 0; x < valid_mask.cols; ++x) {
            if (!mrow[x]) continue;
            *it_x = prow[x][0];
            *it_y = prow[x][1];
            *it_z = prow[x][2];
            ++it_x;
            ++it_y;
            ++it_z;
        }
    }
    return msg;
}

}  // namespace

class PerceptionNode : public rclcpp::Node {
 public:
    PerceptionNode() : Node("s3m_perception") {
        const std::string model_path = declare_parameter("model_path", std::string(""));
        const std::string left_topic = declare_parameter("left_topic", std::string("/stereo/left"));
        const std::string right_topic = declare_parameter("right_topic", std::string("/stereo/right"));
        const double fx = declare_parameter("fx", 700.0);
        const double fy = declare_parameter("fy", 700.0);
        const double cx = declare_parameter("cx", 600.0);
        const double cy = declare_parameter("cy", 180.0);
        const double baseline = declare_parameter("baseline", 0.54);
        const int width = declare_parameter("image_width", 1242);
        const int height = declare_parameter("image_height", 375);

        rig_ = StereoRig::fromIntrinsics(fx, fy, cx, cy, cv::Size(width, height), baseline);
        matcher_ = StereoMatcher(StereoMatcherParams{});

#if defined(S3M_WITH_ONNX)
        if (!model_path.empty()) {
            OnnxDetector::Options opts;
            opts.model_path = model_path;
            detector_ = std::make_unique<OnnxDetector>(std::move(opts));
            RCLCPP_INFO(get_logger(), "detector: onnx (%s)", model_path.c_str());
        } else {
            detector_ = std::make_unique<NullDetector>();
            RCLCPP_WARN(get_logger(), "no model_path given -- detections/tracks will be empty");
        }
#else
        detector_ = std::make_unique<NullDetector>();
        RCLCPP_WARN(get_logger(), "built without S3M_WITH_ONNX -- detections/tracks will be empty");
#endif

        TrackerParams tp;
        tracker_ = Tracker(tp);

        detections_pub_ = create_publisher<vision_msgs::msg::Detection3DArray>(
            "/perception/detections", rclcpp::QoS(10));
        tracks_pub_ = create_publisher<vision_msgs::msg::Detection3DArray>("/perception/tracks",
                                                                           rclcpp::QoS(10));
        cloud_pub_ =
            create_publisher<sensor_msgs::msg::PointCloud2>("/perception/pointcloud", rclcpp::QoS(10));

        left_sub_.subscribe(this, left_topic);
        right_sub_.subscribe(this, right_topic);
        sync_ = std::make_shared<message_filters::Synchronizer<ApproxSync>>(ApproxSync(10), left_sub_,
                                                                             right_sub_);
        sync_->registerCallback(
            std::bind(&PerceptionNode::onStereoPair, this, std::placeholders::_1, std::placeholders::_2));

        RCLCPP_INFO(get_logger(), "s3m_perception ready: %s + %s -> /perception/{detections,tracks,pointcloud}",
                   left_topic.c_str(), right_topic.c_str());
    }

 private:
    void onStereoPair(const Image::ConstSharedPtr& left_msg, const Image::ConstSharedPtr& right_msg) {
        cv::Mat left;
        cv::Mat right;
        try {
            left = cv_bridge::toCvShare(left_msg, "bgr8")->image;
            right = cv_bridge::toCvShare(right_msg, "bgr8")->image;
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(get_logger(), "cv_bridge conversion failed: %s", e.what());
            return;
        }

        const cv::Mat disparity = matcher_.computeDisparity(left, right);
        const cv::Mat depth = disparityToDepthMap(disparity, rig_);
        const std::vector<Detection2D> dets2d = detector_->detect(left);
        const std::vector<Detection3D> dets3d = promoteTo3D(dets2d, depth, rig_);

        detections_pub_->publish(toDetectionArray(dets3d, left_msg->header));

        const TrackerUpdateResult tr = tracker_.update(dets3d);
        std::vector<Detection3D> confirmed;
        std::vector<int> ids;
        for (const TrackState& t : tr.tracks) {
            if (!t.confirmed) continue;
            Detection3D d;
            d.position = t.position;
            d.class_id = t.class_id;
            d.score = 1.0f;
            confirmed.push_back(d);
            ids.push_back(t.id);
        }
        tracks_pub_->publish(toDetectionArray(confirmed, left_msg->header, &ids));

        cv::Mat valid_mask;
        const cv::Mat cloud = reproject(disparity, rig_, &valid_mask);
        cloud_pub_->publish(toPointCloud2(cloud, valid_mask, left_msg->header));
    }

    StereoRig rig_;
    StereoMatcher matcher_;
    std::unique_ptr<Detector> detector_;
    Tracker tracker_;

    message_filters::Subscriber<Image> left_sub_;
    message_filters::Subscriber<Image> right_sub_;
    std::shared_ptr<message_filters::Synchronizer<ApproxSync>> sync_;

    rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr detections_pub_;
    rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr tracks_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PerceptionNode>());
    rclcpp::shutdown();
    return 0;
}

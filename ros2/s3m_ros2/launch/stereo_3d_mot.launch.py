"""Launches s3m_ros2's perception node with configurable topics/calibration.

    ros2 launch s3m_ros2 stereo_3d_mot.launch.py model_path:=/models/yolov8n.onnx
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument("model_path", default_value=""),
        DeclareLaunchArgument("left_topic", default_value="/stereo/left"),
        DeclareLaunchArgument("right_topic", default_value="/stereo/right"),
        DeclareLaunchArgument("fx", default_value="700.0"),
        DeclareLaunchArgument("fy", default_value="700.0"),
        DeclareLaunchArgument("cx", default_value="600.0"),
        DeclareLaunchArgument("cy", default_value="180.0"),
        DeclareLaunchArgument("baseline", default_value="0.54"),
        DeclareLaunchArgument("image_width", default_value="1242"),
        DeclareLaunchArgument("image_height", default_value="375"),
    ]
    node = Node(
        package="s3m_ros2",
        executable="perception_node",
        name="s3m_perception",
        output="screen",
        parameters=[{
            "model_path": LaunchConfiguration("model_path"),
            "left_topic": LaunchConfiguration("left_topic"),
            "right_topic": LaunchConfiguration("right_topic"),
            "fx": LaunchConfiguration("fx"),
            "fy": LaunchConfiguration("fy"),
            "cx": LaunchConfiguration("cx"),
            "cy": LaunchConfiguration("cy"),
            "baseline": LaunchConfiguration("baseline"),
            "image_width": LaunchConfiguration("image_width"),
            "image_height": LaunchConfiguration("image_height"),
        }],
    )
    return LaunchDescription(args + [node])

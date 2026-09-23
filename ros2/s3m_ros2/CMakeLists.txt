cmake_minimum_required(VERSION 3.16)
project(s3m_ros2)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# Pull in the existing, already-tested s3m library from the main project
# rather than reimplementing or re-linking against an installed copy --
# there is exactly one implementation of the pipeline, this package is only
# a ROS2-shaped front door onto it. Apps/tests/warnings-as-errors are
# switched off here: this build only needs the library target, and colcon's
# own build already gives every package its own warning/lint pass.
set(S3M_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(S3M_BUILD_APPS OFF CACHE BOOL "" FORCE)
set(S3M_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)

# Explicit, not a bare "../.." relative to CMAKE_CURRENT_SOURCE_DIR: colcon
# builds this package through a symlink (src/s3m_ros2 -> the real checkout),
# and CMAKE_CURRENT_SOURCE_DIR reflects the symlink's own location in the
# colcon workspace, not the real path it points at -- "../.." from there
# lands in the colcon workspace root, not this repo. S3M_REPO_ROOT overrides
# that when the real path is known (pass -DS3M_REPO_ROOT=/src for this
# project's own Docker-based build); the relative fallback still covers a
# plain, symlink-free checkout built directly.
if(NOT DEFINED S3M_REPO_ROOT)
    set(S3M_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
endif()
add_subdirectory("${S3M_REPO_ROOT}" s3m_core)

add_executable(perception_node src/perception_node.cpp)
target_link_libraries(perception_node PRIVATE
  s3m::s3m
  rclcpp::rclcpp
  cv_bridge::cv_bridge
  ${OpenCV_LIBS})
ament_target_dependencies(perception_node PUBLIC
  sensor_msgs
  vision_msgs
  message_filters)
target_include_directories(perception_node PRIVATE ${OpenCV_INCLUDE_DIRS})

install(TARGETS perception_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

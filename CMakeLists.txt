cmake_minimum_required(VERSION 3.16)

project(stereo_3d_mot
    VERSION 0.1.0
    DESCRIPTION "Real-time stereo vision: depth, 3D localisation and multi-object tracking"
    LANGUAGES CXX)

# --------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------
option(S3M_BUILD_TESTS        "Build unit tests"                     ON)
option(S3M_BUILD_APPS         "Build demo / benchmark executables"   ON)
option(S3M_WARNINGS_AS_ERRORS "Treat compiler warnings as errors"    OFF)

# --------------------------------------------------------------------------
# Global settings
# --------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        Debug Release RelWithDebInfo MinSizeRel)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerWarnings)

# --------------------------------------------------------------------------
# Dependencies
# --------------------------------------------------------------------------
# Prefer OpenCV's CMake package; fall back to pkg-config (Debian/Ubuntu split
# -dev packages ship opencv4.pc but not always OpenCVConfig.cmake).
find_package(OpenCV 4 QUIET COMPONENTS core imgproc calib3d imgcodecs objdetect)
if(OpenCV_FOUND)
    set(S3M_OPENCV_TARGETS ${OpenCV_LIBS})
    set(S3M_OPENCV_INCLUDES ${OpenCV_INCLUDE_DIRS})
    message(STATUS "stereo_3d_mot: OpenCV ${OpenCV_VERSION} (CMake package)")
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(OpenCV REQUIRED IMPORTED_TARGET opencv4)
    set(S3M_OPENCV_TARGETS PkgConfig::OpenCV)
    set(S3M_OPENCV_INCLUDES "")
    message(STATUS "stereo_3d_mot: OpenCV ${OpenCV_VERSION} (pkg-config)")
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
message(STATUS "stereo_3d_mot: Eigen ${Eigen3_VERSION}")

# --------------------------------------------------------------------------
# Core library:  s3m::s3m
# --------------------------------------------------------------------------
add_library(s3m
    src/core/config.cpp
    src/core/timer.cpp
    src/camera/camera_model.cpp
    src/camera/stereo_rig.cpp
    src/depth/stereo_matcher.cpp
    src/depth/depth_metrics.cpp
    src/geometry/reprojection.cpp
    src/io/pfm.cpp
    src/io/point_cloud.cpp
    src/io/synthetic_source.cpp
    src/io/middlebury_source.cpp
    src/detection/detector.cpp
    src/tracking/kalman_filter.cpp
    src/tracking/track.cpp
    src/viz/depth_viz.cpp)

add_library(s3m::s3m ALIAS s3m)

target_include_directories(s3m PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

if(S3M_OPENCV_INCLUDES)
    target_include_directories(s3m SYSTEM PUBLIC ${S3M_OPENCV_INCLUDES})
endif()
target_link_libraries(s3m PUBLIC ${S3M_OPENCV_TARGETS} Eigen3::Eigen)
target_compile_features(s3m PUBLIC cxx_std_17)
s3m_set_target_warnings(s3m)

# --------------------------------------------------------------------------
# Subdirectories
# --------------------------------------------------------------------------
if(S3M_BUILD_APPS)
    add_subdirectory(apps)
endif()

if(S3M_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
message(STATUS "stereo_3d_mot ${PROJECT_VERSION}")
message(STATUS "  build type ......... ${CMAKE_BUILD_TYPE}")
message(STATUS "  build apps ......... ${S3M_BUILD_APPS}")
message(STATUS "  build tests ........ ${S3M_BUILD_TESTS}")
message(STATUS "  warnings as errors . ${S3M_WARNINGS_AS_ERRORS}")

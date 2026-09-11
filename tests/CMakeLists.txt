# GoogleTest: use a system install when present, otherwise fetch a pinned copy.
find_package(GTest QUIET)

if(NOT GTest_FOUND AND NOT TARGET GTest::gtest)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

add_executable(s3m_tests
    test_camera_model.cpp
    test_stereo_rig.cpp
    test_reprojection.cpp
    test_depth_metrics.cpp
    test_stereo_matcher.cpp
    test_pfm.cpp
    test_synthetic_source.cpp
    test_kalman_filter.cpp
    test_config.cpp
    test_letterbox.cpp
    test_nms.cpp
    test_assignment.cpp
    test_tracker.cpp
    test_trajectory_io.cpp
    test_mot_metrics.cpp
    test_kitti_loader.cpp)

if(S3M_ONNX_ENABLED)
    # Copy the ONNX fixture next to the test binary so the suite is relocatable
    # (works from the build tree with no source checkout mounted).
    file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/data/tiny_yolov8.onnx"
         DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/data")
    target_sources(s3m_tests PRIVATE test_onnx_detector.cpp)
    target_compile_definitions(s3m_tests PRIVATE
        "S3M_TEST_DATA_DIR=\"${CMAKE_CURRENT_BINARY_DIR}/data\"")
endif()

target_link_libraries(s3m_tests PRIVATE s3m::s3m GTest::gtest GTest::gtest_main)
s3m_set_target_warnings(s3m_tests)

include(GoogleTest)
gtest_discover_tests(s3m_tests PROPERTIES LABELS "unit")

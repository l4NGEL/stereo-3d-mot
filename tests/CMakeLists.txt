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
    test_config.cpp)

target_link_libraries(s3m_tests PRIVATE s3m::s3m GTest::gtest GTest::gtest_main)
s3m_set_target_warnings(s3m_tests)

include(GoogleTest)
gtest_discover_tests(s3m_tests PROPERTIES LABELS "unit")

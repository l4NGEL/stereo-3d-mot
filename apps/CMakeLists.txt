# Shared app glue (CLI parsing, frame-source factory).
add_library(app_common STATIC common.cpp)
target_include_directories(app_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(app_common PUBLIC s3m::s3m)
s3m_set_target_warnings(app_common)

function(s3m_add_app name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE app_common)
    s3m_set_target_warnings(${name})
endfunction()

s3m_add_app(stereo_depth_demo)
s3m_add_app(benchmark_depth)

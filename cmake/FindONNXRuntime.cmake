# Locate a prebuilt ONNX Runtime (the microsoft/onnxruntime release tarball
# layout: <root>/include/*.h, <root>/lib/libonnxruntime.so).
#
# Search order:  ONNXRUNTIME_ROOT (CMake var or environment), then /opt/onnxruntime,
# then the usual system prefixes.
#
# On success defines:
#   ONNXRuntime_FOUND
#   ONNXRuntime_INCLUDE_DIRS
#   ONNXRuntime_LIBRARIES
#   target  onnxruntime::onnxruntime   (library only; add includes as SYSTEM)

if(TARGET onnxruntime::onnxruntime)
    set(ONNXRuntime_FOUND TRUE)
    return()
endif()

set(_ort_hints "")
if(DEFINED ONNXRUNTIME_ROOT)
    list(APPEND _ort_hints "${ONNXRUNTIME_ROOT}")
endif()
if(DEFINED ENV{ONNXRUNTIME_ROOT})
    list(APPEND _ort_hints "$ENV{ONNXRUNTIME_ROOT}")
endif()
list(APPEND _ort_hints /opt/onnxruntime /usr/local /usr)

find_path(ONNXRuntime_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS ${_ort_hints}
    PATH_SUFFIXES include include/onnxruntime include/onnxruntime/core/session)

find_library(ONNXRuntime_LIBRARY
    NAMES onnxruntime
    HINTS ${_ort_hints}
    PATH_SUFFIXES lib lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNXRuntime
    REQUIRED_VARS ONNXRuntime_LIBRARY ONNXRuntime_INCLUDE_DIR)

if(ONNXRuntime_FOUND)
    set(ONNXRuntime_INCLUDE_DIRS "${ONNXRuntime_INCLUDE_DIR}")
    set(ONNXRuntime_LIBRARIES "${ONNXRuntime_LIBRARY}")

    add_library(onnxruntime::onnxruntime UNKNOWN IMPORTED)
    set_target_properties(onnxruntime::onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRuntime_LIBRARY}")
endif()

mark_as_advanced(ONNXRuntime_INCLUDE_DIR ONNXRuntime_LIBRARY)

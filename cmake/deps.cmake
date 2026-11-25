# Third-party dependencies acquired once for every consumer in this tree.
#
# This file must be included BEFORE add_subdirectory(src) (and any tools/ target
# that links these dependencies). It is the only place that downloads anything;
# libraries and tools only resolve the resulting targets.

include(FetchContent)

if(CMAKE_VERSION VERSION_LESS 3.28)
  message(FATAL_ERROR
    "cmake/deps.cmake needs CMake >= 3.28 for FetchContent EXCLUDE_FROM_ALL/SYSTEM.\n"
    "CMake ${CMAKE_VERSION} was found.")
endif()

set(JSON_BuildTests OFF)   # nlohmann/json otherwise enables its own test suite

FetchContent_Declare(json
  GIT_REPOSITORY https://github.com/nlohmann/json
  GIT_TAG        v3.12.0        # replace with a commit SHA when bumping
  GIT_SHALLOW    TRUE
  EXCLUDE_FROM_ALL
  SYSTEM)
FetchContent_MakeAvailable(json)

set(MINJA_TEST_ENABLED     OFF)
set(MINJA_EXAMPLE_ENABLED  OFF)
set(MINJA_FUZZTEST_ENABLED OFF)
set(MINJA_USE_VENV         OFF)

FetchContent_Declare(minja
  GIT_REPOSITORY https://github.com/google/minja
  GIT_TAG        021c2293c187789ef13d56c6cfd89c9b134fd80f  # main, 2025-09-22
  EXCLUDE_FROM_ALL
  SYSTEM)
FetchContent_MakeAvailable(minja)

# ---------------------------------------------------------------------------
# prometheus-cpp (core only)
#
# No exposer (pull/CivetWeb) and no Pushgateway client (push/libcurl): the
# library only collects, the consumer decides how /metrics is served.
# ---------------------------------------------------------------------------
set(ENABLE_PULL        OFF)
set(ENABLE_PUSH        OFF)
set(ENABLE_COMPRESSION OFF)
set(ENABLE_TESTING     OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # core is static; zinferlm is SHARED

FetchContent_Declare(prometheus_cpp
  GIT_REPOSITORY https://github.com/jupp0r/prometheus-cpp
  GIT_TAG        v1.3.0        # = e5fada43131d251e9c4786b04263ce98b6767ba5
  GIT_SHALLOW    TRUE
  EXCLUDE_FROM_ALL
  SYSTEM)
FetchContent_MakeAvailable(prometheus_cpp)

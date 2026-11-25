cmake_minimum_required(VERSION 3.20...3.25) 

project("inferlm" LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(BUILD_ALL TRUE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

option(BUILD_CLI "Build ZInferLM CLI" ${BUILD_ALL})
option(BUILD_DEBUG "Build ZInferLM debug tool" ${BUILD_ALL})
option(BUILD_TOKENIZE "Build ZInferLM tokenize tool" ${BUILD_ALL})
option(BUILD_SERVER "Build ZinferLM Server" ${BUILD_ALL})
option(BUILD_MEMPLAN "Build ZInferLM memory planner" ${BUILD_ALL})

option(ENABLE_ASAN "Enable Address Sanitizer" OFF)

if (ENABLE_ASAN)
  message(STATUS "AddressSanitizer is enabled")
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=address)
endif()

set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# Default to a configured build type; minja's CMake otherwise force-sets one in
# the global cache, and an empty build type silently disables optimisation.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

# All third-party acquisition lives here; must stay above any consumer.
include(cmake/deps.cmake)

add_subdirectory(ggml)
add_subdirectory(src)


if(BUILD_DEBUG)
  add_subdirectory(tools/debug)
endif()

if(BUILD_TOKENIZE)
  add_subdirectory(tools/tokenize)
endif()

if (BUILD_CLI)
  add_subdirectory(tools/cli)
endif()

if (BUILD_SERVER)
  add_subdirectory(tools/serve)
endif()

if (BUILD_MEMPLAN)
  add_subdirectory(tools/memplan)
endif()

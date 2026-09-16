cmake_minimum_required(VERSION 3.20...3.25) 

project("inferlm" C CXX)

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

set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(ggml)

add_subdirectory(src)

if(BUILD_CLI)
  add_subdirectory(tools/cli)
endif()

if(BUILD_DEBUG)
  add_subdirectory(tools/debug)
endif()

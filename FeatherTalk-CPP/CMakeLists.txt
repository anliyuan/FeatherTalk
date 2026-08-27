cmake_minimum_required(VERSION 3.16)
project(FeatherTalkCpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MNN_ROOT "" CACHE PATH "MNN source or installation root")
find_path(MNN_INCLUDE_DIR MNN/Interpreter.hpp HINTS "${MNN_ROOT}/include")
find_library(MNN_LIBRARY NAMES MNN HINTS
  "${MNN_ROOT}" "${MNN_ROOT}/lib" "${MNN_ROOT}/build"
  "${MNN_ROOT}/build-macos-arm64")
if(NOT MNN_INCLUDE_DIR OR NOT MNN_LIBRARY)
  message(FATAL_ERROR "MNN not found; set -DMNN_ROOT=/path/to/MNN")
endif()

add_library(feathertalk_runtime STATIC src/main.cc)
target_compile_definitions(feathertalk_runtime PRIVATE FEATHERTALK_NO_MAIN)
target_include_directories(feathertalk_runtime
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE "${MNN_INCLUDE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb")
target_link_libraries(feathertalk_runtime PUBLIC "${MNN_LIBRARY}")

add_executable(feathertalk_mnn src/cli_main.cc)
target_link_libraries(feathertalk_mnn PRIVATE feathertalk_runtime)

add_executable(benchmark_mnn src/benchmark_mnn.cc)
target_include_directories(benchmark_mnn PRIVATE "${MNN_INCLUDE_DIR}")
target_link_libraries(benchmark_mnn PRIVATE "${MNN_LIBRARY}")

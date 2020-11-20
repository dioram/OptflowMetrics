cmake_minimum_required(VERSION 3.15)
project(KITTY_METRICS)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake ${CMAKE_MODULE_PATH})

add_subdirectory(3rdparty)
add_subdirectory(readers)
add_subdirectory(optflow_metrics)
add_subdirectory(pytorch)
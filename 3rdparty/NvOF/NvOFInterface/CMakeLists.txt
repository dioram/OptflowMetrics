cmake_minimum_required(VERSION 3.9)
project(NvOFInterface)

add_library(NvOFInterface INTERFACE)
target_include_directories(NvOFInterface INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
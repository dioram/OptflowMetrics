cmake_minimum_required(VERSION 3.15)
project(KittiDevkit)

add_library(KittiDevkit INTERFACE)
target_include_directories(KittiDevkit INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/headers>)

target_link_libraries(KittiDevkit INTERFACE png++)
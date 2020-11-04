cmake_minimum_required(VERSION 3.15)
project(KittiDevkitExe)
set(${PROJECT_NAME}_TARGET_NAME ${PROJECT_NAME})

set(SOURCE_FILES src/main.cpp)

add_executable(${${PROJECT_NAME}_TARGET_NAME} ${SOURCE_FILES})
target_link_libraries(${${PROJECT_NAME}_TARGET_NAME} PRIVATE KittiDevkit)
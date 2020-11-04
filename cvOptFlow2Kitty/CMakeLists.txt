cmake_minimum_required(VERSION 3.15)
project(OpticalFlowKITTY)
set(${PROJECT_NAME}_TARGET_NAME ${PROJECT_NAME})

set(SOURCE_FILES src/main.cpp)

add_executable(${${PROJECT_NAME}_TARGET_NAME} ${SOURCE_FILES})
target_include_directories(${${PROJECT_NAME}_TARGET_NAME}
	PRIVATE
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)

find_package(OpenCV REQUIRED)
target_include_directories(${${PROJECT_NAME}_TARGET_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${${PROJECT_NAME}_TARGET_NAME} PRIVATE ${OpenCV_LIBRARIES})

target_link_libraries(${${PROJECT_NAME}_TARGET_NAME} PRIVATE KittiDevkit)
cmake_minimum_required(VERSION 3.15)
project(readers)
set(${PROJECT_NAME}_TARGET_NAME ${PROJECT_NAME})

set(SOURCE_FILES
	src/CintelReader.h src/CintelReader.cpp
	src/KittiReader.h src/KittiReader.cpp
	src/Readers.cpp)
add_library(${${PROJECT_NAME}_TARGET_NAME} ${SOURCE_FILES})
target_include_directories(${${PROJECT_NAME}_TARGET_NAME}
	PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/headers>
	PRIVATE
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)

set_target_properties(${${PROJECT_NAME}_TARGET_NAME} PROPERTIES 
	CXX_STANDARD 17
	CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)
target_include_directories(${${PROJECT_NAME}_TARGET_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${${PROJECT_NAME}_TARGET_NAME} PRIVATE ${OpenCV_LIBRARIES})

find_package(Boost REQUIRED filesystem)
target_include_directories(${${PROJECT_NAME}_TARGET_NAME} PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(${${PROJECT_NAME}_TARGET_NAME} PRIVATE ${Boost_LIBRARIES})

target_link_libraries(${${PROJECT_NAME}_TARGET_NAME} PRIVATE KittiDevkit)
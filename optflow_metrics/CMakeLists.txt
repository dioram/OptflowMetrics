cmake_minimum_required(VERSION 3.15)
project(optflow_metrics)
set(${PROJECT_NAME}_TARGET_NAME ${PROJECT_NAME})

set(SOURCE_FILES 
	src/main.cpp
	src/Adapters.hpp
	src/NvOptFlow20.h src/NvOptFlow20.cpp)

add_executable(${${PROJECT_NAME}_TARGET_NAME} ${SOURCE_FILES})
target_include_directories(${${PROJECT_NAME}_TARGET_NAME}
	PRIVATE
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)

if (WIN32)
    add_custom_command(TARGET ${${PROJECT_NAME}_TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_PROPERTY:FreeImage,IMPORTED_LOCATION>
        $<TARGET_FILE_DIR:${${PROJECT_NAME}_TARGET_NAME}>)
endif()

find_package(OpenCV REQUIRED)
target_include_directories(${${PROJECT_NAME}_TARGET_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${${PROJECT_NAME}_TARGET_NAME} PRIVATE ${OpenCV_LIBRARIES})

target_link_libraries(${${PROJECT_NAME}_TARGET_NAME} PRIVATE readers NvOFCommon)
cmake_minimum_required(VERSION 3.15)
project(optflow_metrics)
set(${PROJECT_NAME}_TARGET_NAME ${PROJECT_NAME})

set(SOURCE_FILES 
	src/main.cpp
	src/Adapters.hpp
	src/NvOptFlow20.h src/NvOptFlow20.cpp
	src/raftOptFlow.h src/raftOptFlow.cpp
    src/DDFlow.h src/DDFlow.cpp
)

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

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(${${PROJECT_NAME}_TARGET_NAME} PRIVATE "${TORCH_LIBRARIES}")
set_property(TARGET ${${PROJECT_NAME}_TARGET_NAME} PROPERTY CXX_STANDARD 17)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET ${${PROJECT_NAME}_TARGET_NAME}
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:${${PROJECT_NAME}_TARGET_NAME}>
                     COMMAND ${CMAKE_COMMAND} -E copy_directory
                     ${CMAKE_CURRENT_SOURCE_DIR}/models
                     $<TARGET_FILE_DIR:${${PROJECT_NAME}_TARGET_NAME}>/models)
endif (MSVC)

find_package(TensorRT REQUIRED)
target_link_libraries(${${PROJECT_NAME}_TARGET_NAME} PRIVATE ${TensorRT_LIBRARIES})
target_include_directories(${${PROJECT_NAME}_TARGET_NAME} PRIVATE ${TensorRT_INCLUDE_DIR})
#message(FATAL_ERROR "${TensorRT_LIBRARIES}")
if (MSVC)
  add_custom_command(TARGET ${${PROJECT_NAME}_TARGET_NAME}
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TensorRT_RUNTIME_LIBRARIES}
                     $<TARGET_FILE_DIR:${${PROJECT_NAME}_TARGET_NAME}>)
endif()

 include(${onnxruntime_ROOT}/onnxruntime.cmake)
 target_link_libraries(${${PROJECT_NAME}_TARGET_NAME} PRIVATE onnxruntime)
 if (MSVC)
   add_custom_command(TARGET ${${PROJECT_NAME}_TARGET_NAME}
                      POST_BUILD
                      COMMAND ${CMAKE_COMMAND} -E copy_if_different
                      $<TARGET_FILE:onnxruntime>
                      $<TARGET_FILE_DIR:${${PROJECT_NAME}_TARGET_NAME}>)
 endif()

find_package(Boost REQUIRED chrono)
target_include_directories(${${PROJECT_NAME}_TARGET_NAME} PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(${${PROJECT_NAME}_TARGET_NAME} PRIVATE ${Boost_LIBRARIES})
cmake_minimum_required(VERSION 3.9)
project(Common C CXX CUDA)
set(${PROJECT_NAME}_TARGET_NAME NvOFCommon)

include(NvOFBase/source_files.cmake)
include(Utils/source_files.cmake)

add_library(${${PROJECT_NAME}_TARGET_NAME} ${SOURCE_FILES})

target_include_directories(${${PROJECT_NAME}_TARGET_NAME} PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/NvOFBase
    ${CMAKE_CURRENT_SOURCE_DIR}/Utils)
set_target_properties(${${PROJECT_NAME}_TARGET_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Do what the new package does
find_library(CUDA_DRIVER_LIBRARY
             NAMES cuda_driver cuda
             HINTS ${CUDA_TOOLKIT_ROOT_DIR}
                   ENV CUDA_PATH
             PATH_SUFFIXES nvidia/current lib64 lib/x64 lib)
if (NOT CUDA_DRIVER_LIBRARY)
    # Don't try any stub directories until we have exhausted all other search locations.
    find_library(CUDA_DRIVER_LIBRARY
                 NAMES cuda_driver cuda
                 HINTS ${CUDA_TOOLKIT_ROOT_DIR}
                       ENV CUDA_PATH
                 PATH_SUFFIXES lib64/stubs lib/x64/stubs lib/stubs stubs)
endif ()
mark_as_advanced(CUDA_DRIVER_LIBRARY)

add_subdirectory(External/FreeImage)
get_target_property(TEMP FreeImage INTERFACE_INCLUDE_DIRECTORIES)
target_link_libraries(${${PROJECT_NAME}_TARGET_NAME} PUBLIC FreeImage NvOFInterface)
target_include_directories(${${PROJECT_NAME}_TARGET_NAME} PUBLIC ${FREEIMAGE_INC_DIR})
target_link_libraries(${${PROJECT_NAME}_TARGET_NAME} PUBLIC ${CMAKE_DL_LIBS})

find_package(CUDA REQUIRED)
target_include_directories(${${PROJECT_NAME}_TARGET_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})
target_link_libraries(${${PROJECT_NAME}_TARGET_NAME} PUBLIC ${CUDA_LIBRARIES} ${CUDA_DRIVER_LIBRARY})
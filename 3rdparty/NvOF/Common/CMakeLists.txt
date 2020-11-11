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

find_package(CUDA)
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_70,code=\"sm_70,compute_70\")
if ( CMAKE_COMPILER_IS_GNUCC )
    if(NOT "${CUDA_NVCC_FLAGS}" MATCHES "-std=c\\+\\+11" )
        list(APPEND CUDA_NVCC_FLAGS -std=c++11)
    endif()
endif()

#message("cuda lib: ${CUDA_CUDA_LIBRARY}")
#message("cuda include: ${CUDA_INCLUDE_DIRS}")

add_subdirectory(External/FreeImage)
get_target_property(TEMP FreeImage INTERFACE_INCLUDE_DIRECTORIES)
target_link_libraries(${${PROJECT_NAME}_TARGET_NAME} PUBLIC FreeImage NvOFInterface)

target_include_directories(${${PROJECT_NAME}_TARGET_NAME} PUBLIC 
	${CUDA_INCLUDE_DIRS}
	${FREEIMAGE_INC_DIR}
)

target_link_libraries(${${PROJECT_NAME}_TARGET_NAME} PUBLIC ${CUDA_CUDA_LIBRARY} ${CMAKE_DL_LIBS})
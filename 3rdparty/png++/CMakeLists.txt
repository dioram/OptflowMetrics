cmake_minimum_required(VERSION 3.15)
project(png++)

add_library(png++ INTERFACE)

target_include_directories(png++ INTERFACE 
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/headers>)

target_compile_definitions(png++ INTERFACE -D_USE_MATH_DEFINES)

find_package(libpng CONFIG REQUIRED)
set(libpng_INCLUDE_DIR "" CACHE PATH "path to png include dir")
set(libpng_LIBRARIES "" CACHE STRING "paths to png libs")
message("${libpng_INCLUDE_DIR} ${libpng_LIBRARIES}")
if(LIB_PNG_INCLUDE_DIR STREQUAL "" OR libpng_LIBRARIES STREQUAL "")
	message(FATAL_ERROR "libpng_INCLUDE_DIR and libpng_LIBRARIES must be filled")
endif()

target_include_directories(png++ INTERFACE ${libpng_INCLUDE_DIR})
target_link_libraries(png++ INTERFACE png)
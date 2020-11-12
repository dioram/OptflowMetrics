cmake_minimum_required(VERSION 3.9)
project(FreeImage)

if (WIN32)

    add_library(FreeImage SHARED IMPORTED GLOBAL)

    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(ARCH "x64")
    else() 
        set(ARCH "x32")
    endif()

    set_target_properties(FreeImage PROPERTIES
        IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/${ARCH}/FreeImage.dll"
        IMPORTED_IMPLIB "${CMAKE_CURRENT_SOURCE_DIR}/${ARCH}/FreeImage.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}")

else()

    add_library(FreeImage SHARED IMPORTED GLOBAL)
    find_path(FreeImage_INCLUDE_DIR FreeImage.h PATHS /usr/include/)
    find_library(FreeImage_LIBRARY freeimage PATHS /usr/lib/x86_64-linux-gnu/)

    set_target_properties(FreeImage PROPERTIES
            IMPORTED_LOCATION "${FreeImage_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${FreeImage_INCLUDE_DIR}")

endif()
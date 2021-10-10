cmake_minimum_required(VERSION 3.14)

project(irlba
    VERSION 1.0.0
    DESCRIPTION "C++ port of the IRLBA algorithm"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_library(irlba INTERFACE)

target_include_directories(irlba INTERFACE include/)

add_subdirectory(extern)

target_link_libraries(irlba INTERFACE eigen aarand)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
    if(BUILD_TESTING)
        add_subdirectory(tests)
    endif()
endif()

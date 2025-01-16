cmake_minimum_required(VERSION 3.14)

project(sparse-perf
    VERSION 1.0.0
    DESCRIPTION "Performance tests for sparse matrices"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

FetchContent_Declare(
  cli11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11
  GIT_TAG v2.4.1
)

FetchContent_Declare(
    nanobench
    GIT_REPOSITORY https://github.com/martinus/nanobench
    GIT_TAG v4.3.11
)

FetchContent_MakeAvailable(cli11)
FetchContent_MakeAvailable(nanobench)

add_subdirectory(../../extern dependencies)

add_executable(test test.cpp)
target_link_libraries(test CLI11::CLI11 nanobench::nanobench Eigen3::Eigen ltla::aarand ltla::subpar)
target_compile_options(test PRIVATE "-I${CMAKE_CURRENT_SOURCE_DIR}/../../include")

find_package(OpenMP)
if(OpenMP_FOUND)
    target_link_libraries(test OpenMP::OpenMP_CXX)
endif()

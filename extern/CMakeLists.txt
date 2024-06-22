cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

include(FetchContent)

FetchContent_Declare(
  eigen 
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen
  GIT_TAG 3.4.0
)

FetchContent_MakeAvailable(eigen)

FetchContent_Declare(
  aarand
  GIT_REPOSITORY https://github.com/LTLA/aarand
  GIT_TAG master # ^1.0.0
)

FetchContent_MakeAvailable(aarand)

cmake_minimum_required(VERSION 3.14...3.25)

project(irlba
    VERSION 2.0.2
    DESCRIPTION "C++ port of the IRLBA algorithm"
    LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Library
add_library(irlba INTERFACE)
add_library(ltla::irlba ALIAS irlba)

target_include_directories(irlba INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ltla>)
target_compile_features(irlba INTERFACE cxx_std_17)

# Dependencies
option(IRLBA_FETCH_EXTERN "Automatically fetch CppIrlba's dependencies." ON)
if(IRLBA_FETCH_EXTERN)
    add_subdirectory(extern)
else()
    find_package(Eigen3 5.0.0 CONFIG REQUIRED)
    find_package(ltla_aarand 1.0.0 CONFIG REQUIRED)
    find_package(ltla_subpar 0.3.1 CONFIG REQUIRED)
endif()

target_link_libraries(irlba INTERFACE Eigen3::Eigen ltla::aarand ltla::subpar)

# Tests
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    option(IRLBA_TESTS "Build CppIrlba's test suite." ON)
else()
    option(IRLBA_TESTS "Build CppIrlba's test suite." OFF)
endif()
if(IRLBA_TESTS)
    include(CTest)
    if(BUILD_TESTING)
        add_subdirectory(tests)
    endif()
endif()

# Install
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ltla)

install(TARGETS irlba
    EXPORT irlbaTargets)

install(EXPORT irlbaTargets
    FILE ltla_irlbaTargets.cmake
    NAMESPACE ltla::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ltla_irlba)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/ltla_irlbaConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ltla_irlba)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ltla_irlbaConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ltla_irlbaConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/ltla_irlbaConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ltla_irlba)

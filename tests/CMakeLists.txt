include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/afd902e992b720d1b3e106bc5e425a5768872265.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Avoid installing GoogleTest when installing this project.
option(INSTALL_GTEST "Enable installation of googletest." OFF)

FetchContent_MakeAvailable(googletest)

enable_testing()

include(GoogleTest)

option(CODE_COVERAGE "Enable coverage testing" OFF)
set(DO_CODE_COVERAGE OFF)
if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(DO_CODE_COVERAGE ON)
endif()

macro(decorate_test target)
    target_link_libraries(
        ${target}
        gtest_main
        irlba
    )

    target_compile_options(${target} PRIVATE -Wall -Wpedantic -Wextra)

    if(DO_CODE_COVERAGE)
        target_compile_options(${target} PRIVATE -O0 -g --coverage)
        target_link_options(${target} PRIVATE --coverage)
    endif()

    gtest_discover_tests(${target})
endmacro()

add_executable(
    libtest 
    src/utils.cpp
    src/lanczos.cpp
    src/invariant.cpp
    src/compute.cpp
    src/pca.cpp
    #src/sparse.cpp
    #src/MockMatrix.cpp
    #src/wrappers.cpp
    #src/parallel.cpp
)
decorate_test(libtest)

# Check that the custom parallelization schemes are properly set up.
add_executable(
    custom_parallel
    src/parallel.cpp
)
decorate_test(custom_parallel)
target_compile_definitions(custom_parallel PRIVATE TEST_CUSTOM_PARALLEL)

# Check that OpenMP works as expected.
find_package(OpenMP)
if (OpenMP_FOUND)
    add_executable(
        omptest 
        src/parallel.cpp
    )
    decorate_test(omptest)
    target_link_libraries(omptest OpenMP::OpenMP_CXX)

    add_executable(
        scopetest 
        src/EigenThreadScope.cpp
    )
    decorate_test(scopetest)
    target_link_libraries(scopetest OpenMP::OpenMP_CXX)

    add_executable(
        scopetest_custom
        src/EigenThreadScope.cpp
    )
    decorate_test(scopetest_custom)
    target_link_libraries(scopetest_custom OpenMP::OpenMP_CXX)
    target_compile_definitions(scopetest_custom PRIVATE IRLBA_CUSTOM_PARALLEL)

    add_executable(
        scopetest_noomp
        src/EigenThreadScope.cpp
    )
    decorate_test(scopetest_noomp)
    target_link_libraries(scopetest_noomp OpenMP::OpenMP_CXX)
    target_compile_definitions(scopetest_noomp PRIVATE SUBPAR_NO_OPENMP_SIMPLE)

    add_executable(
        scopetest_custom_omp
        src/EigenThreadScope.cpp
    )
    decorate_test(scopetest_custom_omp)
    target_link_libraries(scopetest_custom_omp OpenMP::OpenMP_CXX)
    target_compile_definitions(scopetest_custom_omp PRIVATE IRLBA_CUSTOM_PARALLEL IRLBA_CUSTOM_PARALLEL_USES_OPENMP)
endif()

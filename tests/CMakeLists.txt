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

add_executable(
    libtest 
    src/utils.cpp
    src/lanczos.cpp
    src/irlba.cpp
    src/sparse.cpp
    src/invariant.cpp
    src/MockMatrix.cpp
    src/wrappers.cpp
    src/parallel.cpp
)

target_link_libraries(
    libtest
    gtest_main
    irlba
)

target_compile_options(libtest PRIVATE -Wall -Wpedantic -Wextra)

set(CODE_COVERAGE "Enable coverage testing" OFF)
set(DO_CODE_COVERAGE OFF)
if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(DO_CODE_COVERAGE ON)
    target_compile_options(libtest PRIVATE -O0 -g --coverage)
    target_link_options(libtest PRIVATE --coverage)
endif()

include(GoogleTest)
gtest_discover_tests(libtest)

# Check that the custom parallelization schemes are properly set up.
add_executable(
    custom_parallel
    src/parallel.cpp
)

target_compile_definitions(custom_parallel PRIVATE TEST_CUSTOM_PARALLEL)
target_compile_options(custom_parallel PRIVATE -Wall -Wpedantic -Wextra)

target_link_libraries(
    custom_parallel
    gtest_main
    irlba
)

if(DO_CODE_COVERAGE)
    target_compile_options(custom_parallel PRIVATE -O0 -g --coverage)
    target_link_options(custom_parallel PRIVATE --coverage)
endif()

gtest_discover_tests(custom_parallel)

# Check that OpenMP works as expected.
find_package(OpenMP)
if (OpenMP_FOUND)
    add_executable(
        omptest 
        src/parallel.cpp
    )

    target_link_libraries(
        omptest
        gtest_main
        irlba
    )

    target_compile_options(omptest PRIVATE -Wall -Wpedantic -Wextra)
    target_link_libraries(omptest OpenMP::OpenMP_CXX)
    gtest_discover_tests(omptest)

    if(DO_CODE_COVERAGE)
        target_compile_options(omptest PRIVATE -O0 -g --coverage)
        target_link_options(omptest PRIVATE --coverage)
    endif()
endif()

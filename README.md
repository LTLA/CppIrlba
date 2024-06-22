# C++ library for IRLBA

![Unit tests](https://github.com/LTLA/CppIrlba/actions/workflows/run-tests.yaml/badge.svg)
![Documentation](https://github.com/LTLA/CppIrlba/actions/workflows/doxygenate.yaml/badge.svg)
![Irlba comparison](https://github.com/LTLA/CppIrlba/actions/workflows/compare-irlba.yaml/badge.svg)
[![Codecov](https://codecov.io/gh/LTLA/CppIrlba/branch/master/graph/badge.svg?token=E2AFGW2XDB)](https://codecov.io/gh/LTLA/CppIrlba)

## Overview

This repository contains a header-only C++ library implementing the Augmented Implicitly Restarted Lanczos Bidiagonalization Algorithm (IRLBA) from Baglama and Reichel (2005).
IRLBA is a fast and memory-efficient method for truncated singular value decomposition, and is particularly useful for approximate principal components analysis of large matrices.
The code here is derived from the C code in the [**irlba** R package](https://github.com/bwlewis/irlba), refactored to use the [Eigen](http://eigen.tuxfamily.org/) library for matrix algebra.

## Quick start

Using this library is as simple as including the header file in your source code:

```cpp
#include "irlba/irlba.hpp"

irlba::Options opt;
// optional; specify the workspace, etc.
opt.extra_work = 20;
opt.max_iterations = 50;

// Get the first 5 singular triplets:
auto result = irlba::compute(mat, 5, opt);
result.U; // left singular vectors
result.V; // right singular vectors
result.S; // singular values
```

To perform a PCA:

```cpp
// Get the first 5 principal components without scaling:
auto res = irlba::compute(mat, true, false, 5, opt);
Eigen::MatrixXd components = res.U;
components *= res.S.asDiagonal();
```

See the [reference documentation](https://ltla.github.io/CppIrlba) for more details.

## Building projects

### CMake with `FetchContent`

If you're using CMake, you just need to add something like this to your `CMakeLists.txt`:

```cmake
include(FetchContent)

FetchContent_Declare(
  irlba 
  GIT_REPOSITORY https://github.com/LTLA/CppIrlba
  GIT_TAG master # or any version of interest
)

FetchContent_MakeAvailable(irlba)
```

Then you can link to **irlba** to make the headers available during compilation:

```cmake
# For executables:
target_link_libraries(myexe ltla::irlba)

# For libaries
target_link_libraries(mylib INTERFACE ltla::irlba)
```

### CMake with `find_package()`

```cmake
find_package(ltla_irlba CONFIG REQUIRED)
target_link_libraries(mylib INTERFACE ltla::irlba)
```

To install the library use:

```sh
mkdir build && cd build
cmake .. -DIRLBA_TESTS=OFF
cmake --build . --target install
```

By default, this will use `FetchContent` to fetch all external dependencies.
If you want to install them manually, use `-DPOWERIT_FETCH_EXTERN=OFF`.
See the commit hashes in [`extern/CMakeLists.txt`](extern/CMakeLists.txt) to find compatible versions of each dependency.

### Manual

If you're not using CMake, the simple approach is to just copy the files - either directly or with Git submodules - and include their path during compilation with, e.g., GCC's `-I`.
Note that this requires manual management of a few dependencies:

- [**Eigen**](https://gitlab.com/libeigen/eigen), for matrix manipulations.
- [**aarand**](https://github.com/LTLA/aarand), for system-agnostic random distribution functions.

See [`extern/CMakeLists.txt`](extern/CMakeLists.txt) for more details.

## References

Baglama, James and Reichel, Lothar (2005).
Augmented implicitly restarted Lanczos bidiagonalization methods.
_SIAM J. Sci. Comput._, 27(1), 19-42.


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

// Define an Eigen matrix for the input.
Eigen::MatrixXd mat;

// Get the first 5 singular triplets:
auto result = irlba::compute_simple(mat, 5, opt);
result.U; // left singular vectors
result.V; // right singular vectors
result.D; // singular values
```

To perform a PCA:

```cpp
// Get the first 5 principal components without scaling:
auto pcres = irlba::pca(mat, true, false, 5, opt);
pcres.components;
pcres.rotation;
pcres.variances;
```

See the [reference documentation](https://ltla.github.io/CppIrlba) for more details.

## Customizing matrices

The `irlba::Matrix` interface allows us to use different matrix representations for `irlba::compute()`.
For example, we can defer column-centering to efficiently perform PCA on a sparse matrix:

```cpp
// Some sparse matrix.
Eigen::SparseMatrix<double> spmat;

// Wrap the Eigen matrix in a wrapper for compatibility with irlba::Matrix.
irlba::SimpleMatrix<Eigen::VectorXd, Eigen::MatrixXd, decltype(&spmat)> wrapped(&spmat);

// Define column centers.
Eigen::VectorXd center;

// Create a matrix where the column-centering is performed during multiplication.
// This avoids instantiating the actual centered matrix, which would lose sparsity.
irlba::CenteredMatrix<Eigen::VectorXd, Eigen::MatrixXd> centered(&wrapped, &center);

auto centered_res = irlba::compute(centered, 5, opt);
```

We provide several subclasses that implement the `irlba::Matrix` interface:

- `SimpleMatrix`, which wraps existing **Eigen** matrices.
- `CenteredMatrix`, for deferred centering of columns.
- `ScaledMatrix`, for deferred scaling of rows or columns.
- `ParallelSparseMatrix`, for parallelized multiplication of a sparse matrx.

Developers can easily create their own `Matrix` subclass by implementing methods for matrix-vector multiplication. 
For example, the [**scran_pca**](https://github.com/libscran/scran_pca) library performs PCA on a matrix of residuals without ever explicitly creating that matrix.

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
See [`extern/CMakeLists.txt`](extern/CMakeLists.txt) to find compatible versions of each dependency.

### Manual

If you're not using CMake, the simple approach is to just copy the files - either directly or with Git submodules - and include their path during compilation with, e.g., GCC's `-I`.
Note that this requires the dependencies listed in [`extern/CMakeLists.txt`](extern/CMakeLists.txt). 

## References

Baglama, James and Reichel, Lothar (2005).
Augmented implicitly restarted Lanczos bidiagonalization methods.
_SIAM J. Sci. Comput._, 27(1), 19-42.


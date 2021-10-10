# C++ library for IRLBA

![Unit tests](https://github.com/LTLA/CppIrlba/actions/workflows/run-tests.yaml/badge.svg)
![Documentation](https://github.com/LTLA/CppIrlba/actions/workflows/doxygenate.yaml/badge.svg)
![Irlba comparison](https://github.com/LTLA/CppIrlba/actions/workflows/compare-irlba.yaml/badge.svg)
[![Codecov](https://codecov.io/gh/LTLA/CppIrlba/branch/master/graph/badge.svg?token=E2AFGW2XDB)](https://codecov.io/gh/LTLA/CppIrlba)

## Overview

This repository contains a header-only C++ library implementing the Augmented Implicitly Restarted Lanczos Bidiagonalization Algorithm (IRLBA) from Baglama and Lothar (2005).
IRLBA is a fast and memory-efficient method for truncated singular value decomposition, and is particularly useful for approximate principal components analysis of large matrices.
The code here is derived from the C code in the [**irlba** R package](https://github.com/bwlewis/irlba), refactored to use the [Eigen](http://eigen.tuxfamily.org/) library for matrix algebra.

## Quick start

Using this library is as simple as including the header file in your source code:

```cpp
#include "irlba/irlba.hpp"

irlba::Irlba runner;

// optional; specify the number of singular vectors, workspace, etc.
runner.set_number(5).set_work(20);

auto result = runner.run(mat, false, false, U, V, S);
result.U; // left singular vectors
result.V; // right singular vectors
result.S; // singular values
```

To perform a PCA:

```cpp
auto res = runner.run(mat, true, false);
Eigen::MatrixXd components = res.U;
components *= res.S.asDiagonal();
```

See the [reference documentation](https://ltla.github.io/CppIrlba) for more details.

## References

Baglama, James, and Lothar Reichel (2005).
Augmented implicitly restarted Lanczos bidiagonalization methods.
_SIAM J. Sci. Comput._, 27(1), 19-42.


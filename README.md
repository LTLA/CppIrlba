# C++ library for IRLBA

## Overview

This repository contains a header-only C++ library implementing the Augment Implicitly Restarted Lanczos Bidiagonalization Algorithm (IRLBA) from Baglama and Lothar (2005).
IRLBA is fast and memory-efficient method for truncated singular value decomposition, and is particularly useful for approximate principal components analysis of large matrices.
The code here is derived from the C code in the [**irlba** R package](https://github.com/bwlewis/irlba), refactored to use the [Eigen](http://eigen.tuxfamily.org/) library for matrix algebra.

## Quick start

Using this library is as simple as including the header file in your source code:

```cpp
#include "irlba/irlba.hpp"

irlba::Irlba runner;
irlba::NormalSampler norm(50);

// optional; specify the number of singular vectors, workspace, etc.
runner.set_number(5).set_work(20);

Eigen::MatrixXd U, V;
Eigen::VectorXd S;
runner.run(mat, false, false, norm, U, V, S);
```

To perform a PCA, compute the relevant vector of centering values (and optionally scaling values, if desired) and supply this to `run()`:

```cpp
Eigen::VectorXd center = mat.colwise().sum();
runner.run(mat, center, false, norm, U, V, S);
U *= S.asDiagonal();
```

See the [reference documentation](https://ltla.github.io/CppIrlba) for more details.

## References

Baglama, James, and Lothar Reichel (2005).
Augmented implicitly restarted Lanczos bidiagonalization methods.
_SIAM J. Sci. Comput._, 27(1), 19-42.


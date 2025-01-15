# Comparing our sparse matrix to `Eigen`

The main purpose of `irlba::ParallelSparseMatrix` is to support parallelization schemes other than OpenMP.
Doing so may be necessary in environments without OpenMP (e.g., Emscripten) or when OpenMP interferes with other things (e.g., POSIX forks).
Even so, we would like to know whether the speed of matrix multiplication for `irlba::ParallelSparseMatrix` is comparable to that `Eigen::SpMat`,
as this would allow us to generally choose the former when writing library code that might run everywhere.

To test this, we have a little benchmarking executable that can be created and run with:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
./build/test -r 10000 -c 1000 -d 0.1 -t 4
```

# Results

All simulations use a density of 10% non-zero elements with 10 multiplication iterations.
All times are show in milliseconds per operation

On the Apple M2 running Sonoma 14.6.1 with Clang 15.0.0 (no OpenMP support):

|  Rows | Columns | Threads | Custom | Eigen | Custom (adjoint) | Eigen (adjoint) |
|-------|---------|---------|--------|-------|------------------|-----------------|
| 10000 |   10000 |       1 |    423 |   423 |             1138 |            1117 |
| 10000 |   10000 |       4 |    387 |   425 |              323 |            1119 |
| 50000 |    1000 |       1 |    476 |   479 |              601 |             600 |
| 50000 |    1000 |       4 |    132 |   476 |              171 |             585 |
|  2000 |   10000 |       1 |     83 |    81 |              160 |             159 |
|  2000 |   10000 |       4 |     72 |    82 |               52 |             161 |

# Building the function.
library(Rcpp)
if (!file.exists("irlba")) {
    file.symlink("../../include/irlba", "irlba")
}
sourceCpp("test.cpp")

# Generating some data.
set.seed(10)
mat <- matrix(rnorm(500), 25, 20)
set.seed(100)
v <- rnorm(ncol(mat))

# Creating a comparator function, as decompositions
# are not identifiable by the sign of the vectors.
compare <- function(left, right) {
    expect_equal(abs(left), abs(right))
    expect_equal(abs(colSums(left)), abs(colSums(right)))
}

# Running it against the reference implementation in the R package.
library(irlba)
library(testthat)

test_that("simple", {
    ref <- irlba::irlba(mat, nv=3, work=10, v=v)
    out <- run_irlba(mat, v, number=3, work=7)

    compare(ref$u, out$U)
    compare(ref$v, out$V)
    expect_equal(ref$s, out$d)

    # Another one, just for good measure. 
    ref <- irlba::irlba(mat, nv=10, work=15, v=rev(v))
    out <- run_irlba(mat, rev(v), number=10, work=5)

    compare(ref$u, out$U)
    compare(ref$v, out$V)
    expect_equal(ref$s, out$d)
})

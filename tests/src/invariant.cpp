#include <gtest/gtest.h>
#include "irlba/irlba.hpp"
#include "Eigen/Dense"
#include "NormalSampler.h"
#include "compare.h"

class InvariantTester : public ::testing::TestWithParam<int> {
protected:
    void assemble(size_t rank) {
        A = Eigen::MatrixXd(nr, nc);

        // Making a low rank matrix by just duplicating columns.
        NormalSampler norm(42); 
        for (size_t i = 0; i < rank; ++i) {
            for (size_t j = 0; j < nr; ++j) {
                A(j, i) = norm();
            }
        }

        for (size_t i = rank; i < nc; ++i) {
            for (size_t j = 0; j < nr; ++j) {
                A(j, i) = A(j, i % rank);
            }
        }
    }

    size_t nr = 100, nc = 50;
    Eigen::MatrixXd A;
};

TEST_P(InvariantTester, SingularCheck) {
    int rank = GetParam();
    assemble(rank);
    auto res = irlba::compute(A, rank + 3, irlba::Options());

    // Checking that the first 'rank' columns have non-zero singular values,
    // while the remainders are on zero.    
    for (int r = 0; r < rank; ++r) {
        EXPECT_TRUE(res.D[r] > 0.01);
    }
    for (int r = rank; r < rank + 3; ++r) {
        EXPECT_TRUE(res.D[r] < 1e-8);
    }

    // Comparing the sum of squared singular values to the Frobenius norm;
    // they should be the same.
    double total_var = 0;
    for (size_t r = 0; r < res.D.size(); ++r) {
        total_var += res.D[r] * res.D[r];
    }

    double ref = 0;
    for (Eigen::Index i = 0; i < A.cols(); ++i) {
        Eigen::VectorXd current = A.col(i); 
        for (auto x : current) {
            ref += x*x;
        }
    }
    same_same(total_var, ref, 1e-8);
}

TEST_P(InvariantTester, Recovery) {
    int rank = GetParam();
    assemble(rank);

    auto res = irlba::compute(A, rank + 2, irlba::Options());

    // Check that recovered low-rank matrix is identical to the input.
    Eigen::MatrixXd recovered = res.U * res.D.asDiagonal() * res.V.adjoint();

    ASSERT_EQ(recovered.rows(), A.rows());
    ASSERT_EQ(recovered.cols(), A.cols());

    for (size_t c = 0; c < A.cols(); ++c) {
        for (size_t r = 0; r < A.rows(); ++r) {
            same_same(recovered(r, c), A(r, c), 1e-8);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    InvariantTests,
    InvariantTester,
    ::testing::Values(2, 5, 10) // actual rank of the matrix.
);

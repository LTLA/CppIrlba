#include <gtest/gtest.h>

#include "irlba/irlba.hpp"

#include "compare.h"
#include "NormalSampler.h"

#include "Eigen/Dense"
#include "Eigen/Sparse"

typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

class SparseTester : public ::testing::Test {
protected:
    void SetUp () {
        A = Eigen::MatrixXd(nr, nc);
        A.setZero();
        B = SpMat(nr, nc);

        int counter = 0;
        std::vector<T> coefficients;
        NormalSampler norm(42);
        for (size_t i = 0; i < nc; ++i) {
            for (size_t j = 0; j < nr; ++j, ++counter) {
                if (norm() > 0) { // introducing sparsity by only filling in ~50% elements.
                    A(j, i) = norm();
                    coefficients.push_back(T(j, i, A(j, i)));
                }
            }
        }

        B.setFromTriplets(coefficients.begin(), coefficients.end());
        return;
    }

    size_t nr = 100, nc = 50;
    Eigen::MatrixXd A;
    SpMat B;
};

TEST_F(SparseTester, Sparse) {
    irlba::Irlba irb;
    auto res = irb.set_number(8).set_work(7).run(A);
    auto res2 = irb.set_number(8).set_work(7).run(B);

    expect_equal_vectors(res.D, res2.D);
    expect_equal_column_vectors(res.U, res2.U);
    expect_equal_column_vectors(res.V, res2.V);
}

TEST_F(SparseTester, CenterScale) {
    irlba::Irlba irb;
    auto res = irb.set_number(8).set_work(7).run(A, true, true);
    auto res2 = irb.set_number(8).set_work(7).run(B, true, true);

    expect_equal_vectors(res.D, res2.D);
    expect_equal_column_vectors(res.V, res2.V);

    // Don't compare U, as this will always be zero.
    for (size_t i = 0; i < res.U.cols(); ++i) {
        for (size_t j = 0; j < res.U.rows(); ++j) {
            double labs = std::abs(res.U(j, i));
            double rabs = std::abs(res2.U(j, i));
            EXPECT_TRUE(same_same(labs, rabs, 1e-8));
        }
        EXPECT_TRUE(std::abs(res2.U.col(i).sum()) < 1e-8);
    }    
}

TEST_F(SparseTester, SparseToReference) {
    irlba::Irlba irb;
    auto res = irb.set_number(13).set_work(20).run(B);

    // Bumping up the tolerance as later SV's tend to be a bit more variable.
    Eigen::BDCSVD svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    expect_equal_vectors(res.D, svd.singularValues().head(13), 1e-5);
    expect_equal_column_vectors(res.U, svd.matrixU().leftCols(13), 1e-5);
    expect_equal_column_vectors(res.V, svd.matrixV().leftCols(13), 1e-5);
}

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
    irlba::Options opt;
    opt.extra_work = 7;
    auto res = irlba::compute(A, 8, opt);
    auto res2 = irlba::compute(B, 8, opt);

    expect_equal_vectors(res.D, res2.D);
    expect_equal_column_vectors(res.U, res2.U);
    expect_equal_column_vectors(res.V, res2.V);
}

TEST_F(SparseTester, CenterScale) {
    irlba::Options opt;
    opt.extra_work = 7;
    auto res = irlba::compute(A, true, true, 8, opt);
    auto res2 = irlba::compute(B, true, true, 8, opt);

    expect_equal_vectors(res.D, res2.D);
    expect_equal_column_vectors(res.V, res2.V);

    // Don't compare U, as this will always be zero.
    for (Eigen::Index i = 0; i < res.U.cols(); ++i) {
        for (Eigen::Index j = 0; j < res.U.rows(); ++j) {
            double labs = std::abs(res.U(j, i));
            double rabs = std::abs(res2.U(j, i));
            EXPECT_TRUE(same_same(labs, rabs, 1e-8));
        }
        EXPECT_TRUE(std::abs(res2.U.col(i).sum()) < 1e-8);
    }    
}

TEST_F(SparseTester, SparseToReference) {
    irlba::Options opt;
    opt.extra_work = 20;
    auto res = irlba::compute(B, 13, opt);

    // Bumping up the tolerance as later SV's tend to be a bit more variable.
    Eigen::JacobiSVD svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    expect_equal_vectors(res.D, svd.singularValues().head(13), 1e-5);
    expect_equal_column_vectors(res.U, svd.matrixU().leftCols(13), 1e-5);
    expect_equal_column_vectors(res.V, svd.matrixV().leftCols(13), 1e-5);
}

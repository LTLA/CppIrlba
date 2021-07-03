#include <gtest/gtest.h>

#include "irlba/irlba.hpp"

#include "compare.h"

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
        irlba::NormalSampler norm(42);
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
    irlba::NormalSampler norm(50);
    auto res = irb.set_number(8).set_work(7).run(A, false, false, norm);

    irlba::NormalSampler norm2(50);
    auto res2 = irb.set_number(8).set_work(7).run(B, false, false, norm2);
    
    expect_equal_vectors(res.D, res2.D);
    expect_equal_column_vectors(res.U, res2.U);
    expect_equal_column_vectors(res.V, res2.V);
}
#include <gtest/gtest.h>

#include "compare.h"

#include "irlba/irlba.hpp"
#include "irlba/utils.hpp"

#include "Eigen/Dense"

class IrlbaTester : public ::testing::Test {
protected:
    void SetUp () {
        A = Eigen::MatrixXd(nr, nc);

        irlba::NormalSampler norm(42);
        for (size_t i = 0; i < nc; ++i) {
            for (size_t j = 0; j < nr; ++j) {
                A(j, i) = norm();
            }
        }
    }

    size_t nr = 20, nc = 10;
    Eigen::MatrixXd A;
};

TEST_F(IrlbaTester, Basic) {
    irlba::Irlba irb;
    irlba::NormalSampler norm(50);

    auto res = irb.set_number(5).run(A, norm);
    ASSERT_EQ(res.V.cols(), 5);
    ASSERT_EQ(res.U.cols(), 5);
    ASSERT_EQ(res.D.size(), 5);

    Eigen::BDCSVD svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    expect_equal_vectors(res.D, svd.singularValues().head(5));
    expect_equal_column_vectors(res.U, svd.matrixU().leftCols(5));
    expect_equal_column_vectors(res.V, svd.matrixV().leftCols(5));
}

TEST_F(IrlbaTester, CenterScale) {
    irlba::Irlba irb;
    irlba::NormalSampler norm(50);

    Eigen::VectorXd center(A.cols());
    for (auto& c : center) { c = norm(); }
    Eigen::VectorXd scale(A.cols());
    for (auto& s : scale) { s = std::abs(norm() + 1); }

    auto res = irb.run(A, center, scale, norm);

    Eigen::MatrixXd copy = A;
    for (size_t i = 0; i < A.cols(); ++i) {
        for (size_t j = 0; j < A.rows(); ++j) {
            copy(j, i) -= center(i);
            copy(j, i) /= scale(i);
        }
    }

    irlba::NormalSampler norm2(50);
    auto res2 = irb.run(copy, norm2);

    expect_equal_vectors(res.D, res2.D);
    expect_equal_column_vectors(res.U, res2.U);
    expect_equal_column_vectors(res.V, res2.V);
}

TEST_F(IrlbaTester, CenterScaleAgain) {
    irlba::Irlba irb;

    irlba::NormalSampler norm(50);
    auto ref = irb.run(A, norm);

    irlba::NormalSampler norm2(50);
    auto res2 = irb.run<true>(A, norm2);
    EXPECT_NE(ref.D, res2.D);
    for (size_t i = 0; i < res2.U.cols(); ++i) {
        EXPECT_TRUE(std::abs(res2.U.col(i).sum()) < 1e-8);
    }    

    irlba::NormalSampler norm3(50);
    auto res3 = irb.run<true, true>(A, norm3);
    EXPECT_NE(ref.D, res3.D);
    EXPECT_NE(res2.D, res3.D);
    for (size_t i = 0; i < res3.U.cols(); ++i) {
        EXPECT_TRUE(std::abs(res3.U.col(i).sum()) < 1e-8);
    }    

    irlba::NormalSampler norm4(50);
    auto res4 = irb.run<false, true>(A, norm4);
    EXPECT_NE(ref.D, res4.D);
    EXPECT_NE(res2.D, res4.D);
    EXPECT_NE(res3.D, res4.D);
}

TEST_F(IrlbaTester, Exact) {
    irlba::Irlba irb;
    irlba::NormalSampler norm(50);

    Eigen::MatrixXd small = A.leftCols(3);
    auto res = irb.set_number(2).run(small, norm);
      
    Eigen::BDCSVD svd(small, Eigen::ComputeThinU | Eigen::ComputeThinV);
    EXPECT_EQ(svd.singularValues().head(2), res.D);
    EXPECT_EQ(svd.matrixU().leftCols(2), res.U);
    EXPECT_EQ(svd.matrixV().leftCols(2), res.V);
}

TEST_F(IrlbaTester, ExactCenterScale) {
    irlba::Irlba irb;
    irlba::NormalSampler norm(50);

    Eigen::MatrixXd small = A.leftCols(3);

    Eigen::VectorXd center(small.cols());
    for (auto& c : center) { c = norm(); }
    Eigen::VectorXd scale(small.cols());
    for (auto& s : scale) { s = std::abs(norm() + 1); }

    auto res = irb.set_number(2).run(small, center, scale, norm);

    Eigen::MatrixXd copy = small;
    for (size_t i = 0; i < small.cols(); ++i) {
        for (size_t j = 0; j < small.rows(); ++j) {
            copy(j, i) -= center(i);
            copy(j, i) /= scale(i);
        }
    }

    auto res2 = irb.set_number(2).run(copy, norm);

    EXPECT_EQ(res.U, res2.U);
    EXPECT_EQ(res.V, res2.V);
    EXPECT_EQ(res.D, res2.D);
}

using IrlbaDeathTest = IrlbaTester;

TEST_F(IrlbaDeathTest, AssertionFails) {
    irlba::Irlba irb;
    irlba::NormalSampler norm(50);

    // Requested number of SVs > smaller dimension of the matrix.
    ASSERT_DEATH(irb.set_number(100).run(A, norm), "number");

    // Initialization vector is not of the right length.
    Eigen::VectorXd init(1);
    ASSERT_DEATH(irb.set_number(5).set_init(init).run(A, norm), "initV");
}

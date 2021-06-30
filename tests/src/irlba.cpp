#include <gtest/gtest.h>
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
    Eigen::MatrixXd U, V;
    Eigen::VectorXd S;

    void expect_equal_column_vectors(const Eigen::MatrixXd& left, const Eigen::MatrixXd& right) {
        ASSERT_EQ(left.cols(), right.cols());
        ASSERT_EQ(left.rows(), right.rows());

        for (size_t i = 0; i < left.cols(); ++i) {
            for (size_t j = 0; j < left.rows(); ++j) {
                EXPECT_FLOAT_EQ(std::abs(left(j, i)), std::abs(right(j, i)));
            }
            EXPECT_FLOAT_EQ(std::abs(left.col(i).sum()), std::abs(right.col(i).sum()));
        }
        return;
    }

    void expect_equal_vectors(const Eigen::VectorXd& left, const Eigen::VectorXd& right) {
        ASSERT_EQ(left.size(), right.size());
        for (size_t i = 0; i < left.size(); ++i) {
            EXPECT_FLOAT_EQ(left[i], right[i]);
        }
        return;
    }
};

TEST_F(IrlbaTester, Basic) {
    irlba::Irlba irb;
    irlba::NormalSampler norm(50);

    irb.set_number(5).run(A, false, false, norm, U, V, S);
    ASSERT_EQ(V.cols(), 5);
    ASSERT_EQ(U.cols(), 5);
    ASSERT_EQ(S.size(), 5);

    Eigen::BDCSVD svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    expect_equal_vectors(S, svd.singularValues().head(5));
    expect_equal_column_vectors(U, svd.matrixU().leftCols(5));
    expect_equal_column_vectors(V, svd.matrixV().leftCols(5));
}

TEST_F(IrlbaTester, CenterScale) {
    irlba::Irlba irb;
    irlba::NormalSampler norm(50);

    Eigen::VectorXd center(A.cols());
    for (auto& c : center) { c = norm(); }
    Eigen::VectorXd scale(A.cols());
    for (auto& s : scale) { s = std::abs(norm() + 1); }

    irb.run(A, center, scale, norm, U, V, S);

    Eigen::MatrixXd copy = A;
    for (size_t i = 0; i < A.cols(); ++i) {
        for (size_t j = 0; j < A.rows(); ++j) {
            copy(j, i) -= center(i);
            copy(j, i) /= scale(i);
        }
    }

    irlba::NormalSampler norm2(50);
    Eigen::MatrixXd U2(U.rows(), U.cols());
    Eigen::MatrixXd V2(V.rows(), V.cols());
    Eigen::VectorXd S2(S.size());
    irb.run(copy, false, false, norm2, U2, V2, S2);

    expect_equal_vectors(S, S2);
    expect_equal_column_vectors(U, U2);
    expect_equal_column_vectors(V, V2);
}

TEST_F(IrlbaTester, Exact) {
    irlba::Irlba irb;
    irlba::NormalSampler norm(50);

    Eigen::MatrixXd small = A.leftCols(3);
    irb.set_number(2).run(small, false, false, norm, U, V, S);
      
    Eigen::BDCSVD svd(small, Eigen::ComputeThinU | Eigen::ComputeThinV);
    EXPECT_EQ(svd.singularValues().head(2), S);
    EXPECT_EQ(svd.matrixU().leftCols(2), U);
    EXPECT_EQ(svd.matrixV().leftCols(2), V);
}

TEST_F(IrlbaTester, ExactCenterScale) {
    irlba::Irlba irb;
    irlba::NormalSampler norm(50);

    Eigen::MatrixXd small = A.leftCols(3);

    Eigen::VectorXd center(small.cols());
    for (auto& c : center) { c = norm(); }
    Eigen::VectorXd scale(small.cols());
    for (auto& s : scale) { s = std::abs(norm() + 1); }

    irb.set_number(2).run(small, center, scale, norm, U, V, S);

    Eigen::MatrixXd copy = small;
    for (size_t i = 0; i < small.cols(); ++i) {
        for (size_t j = 0; j < small.rows(); ++j) {
            copy(j, i) -= center(i);
            copy(j, i) /= scale(i);
        }
    }

    Eigen::MatrixXd U2(U.rows(), U.cols());
    Eigen::MatrixXd V2(V.rows(), V.cols());
    Eigen::VectorXd S2(S.size());
    irb.set_number(2).run(copy, false, false, norm, U2, V2, S2);

    EXPECT_EQ(U, U2);
    EXPECT_EQ(V, V2);
    EXPECT_EQ(S, S2);
}

using IrlbaDeathTest = IrlbaTester;

TEST_F(IrlbaDeathTest, AssertionFails) {
    irlba::Irlba irb;
    irlba::NormalSampler norm(50);
    Eigen::MatrixXd U, V;
    Eigen::VectorXd S;

    // Requested number of SVs > smaller dimension of the matrix.
    ASSERT_DEATH(irb.set_number(100).run(A, false, false, norm, U, V, S), "number");

    // Initialization vector is not of the right length.
    Eigen::VectorXd init(1);
    ASSERT_DEATH(irb.set_number(5).set_init(init).run(A, false, false, norm, U, V, S), "initV");
}

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

/* 
 * set.seed(10)
 * mat <- matrix(rnorm(500), 25, 20)
 * mat <- round(mat, 2)
 * set.seed(100)
 * v <- rnorm(20)
 * v <- round(v, 2)
 * irlba::irlba(mat, nv=3, work=10, v=v)$d
 *
 * cat(deparse(as.vector(mat)), sep="\n")
 * cat(deparse(v), sep="\n")
 */

std::vector<double> store {
    0.02, -0.18, -1.37, -0.6, 0.29, 0.39, -1.21, -0.36, -1.63,
    -0.26, 1.1, 0.76, -0.24, 0.99, 0.74, 0.09, -0.95, -0.2, 0.93,
    0.48, -0.6, -2.19, -0.67, -2.12, -1.27, -0.37, -0.69, -0.87,
    -0.1, -0.25, -1.85, -0.08, 0.97, 0.18, -1.38, -1.44, 0.36, -1.76,
    -0.32, -0.65, 1.09, -0.76, -0.83, 0.83, -0.97, -0.03, 0.23, -0.3,
    -0.68, 0.66, -0.4, -0.33, 1.37, 2.14, 0.51, 0.79, -0.9, 0.53,
    -0.65, 0.29, -1.24, -0.46, -0.83, 0.34, 1.07, 1.22, 0.74, -0.48,
    0.56, -1.25, 0.38, -1.43, -1.05, -0.22, -1.49, 1.17, -1.48, -0.43,
    -1.05, 1.52, 0.59, -0.22, 0.71, 0.72, 0.44, 0.16, 0.66, 2.22,
    -1.18, -0.07, -0.42, -0.19, 0.07, 1.16, 0.59, -1.42, -1.61, 0.89,
    0.15, 1.23, -0.76, 0.42, -1.04, 0.71, -0.63, 0.56, 0.66, -1.66,
    1.03, 1.13, -1.28, 1.13, -0.46, -0.32, 0.92, 0.08, 1.04, 0.74,
    1.26, 0.95, -0.48, 0.2, -0.03, -1.2, 0.62, -0.91, 0.25, -1.06,
    -0.36, -1.21, 1.43, 0.63, -2, -0.68, -0.46, -0.98, 0.5, 0.73,
    0.67, 0.95, -1.68, -1.21, -1.96, 1.47, 0.37, 1.07, 0.53, 0.1,
    1.34, 0.09, -0.39, -0.25, 1.16, -0.86, -0.87, -2.32, 0.61, 1.15,
    -1.2, -1.58, 0.65, -0.55, 0.52, -0.7, -0.44, -0.68, 0.96, -1.47,
    0.18, -1.44, -1.14, -0.41, 0.14, 1.06, -0.57, 1.28, 0.23, -0.31,
    0.96, 0.55, 0.43, 0.64, -1.36, -0.2, 0.62, 2.07, -0.31, 0.28,
    0.69, 0.05, 0.11, 1, -0.68, -1.28, -1.47, -0.31, -1.7, -1.35,
    -1.1, -1.1, 1.22, 0.33, 1.39, 0.87, -1.08, 0.5, 1.05, -1.27,
    -0.19, -1.3, 0.14, 1.26, -0.43, -1.82, 0.35, -1.35, 0.71, -0.41,
    -0.45, -1.04, -0.33, -0.28, 0.43, -0.31, -0.06, 0.73, 0.1, 1.63,
    0.56, 1.33, -0.28, -1.27, -0.25, 0.02, 0.38, 0.8, -0.84, -2.21,
    -1.13, -1.34, 1.61, 0.74, 0.86, 0.4, 0.51, -0.12, 0.09, -0.36,
    -0.36, 1.03, 1.08, 0.93, -1.46, -0.91, -0.68, 1.06, -0.69, -1.13,
    -1.09, -1.01, 0.41, 0.48, -2.33, 0.02, 0.98, 0.81, 0.12, -2.44,
    0.03, -0.34, -0.26, -0.36, -0.48, 0.22, 2.43, 1.5, -0.72, -0.47,
    0.66, 2.3, 0.33, 0.06, -1.14, 1.18, 0.04, -1.21, 0.07, -0.26,
    0.27, 1.39, 0.19, 0.59, -0.83, 0.39, 0.38, 1.05, 1.16, -1.03,
    -0.25, 1.27, 1.5, 0.59, -0.63, 0.79, 0.13, 0.32, -0.45, 0.77,
    -1.4, -1.18, 0.51, 1.32, 2.39, -0.07, -0.2, 2, 1.54, -1.23, 2.64,
    0.76, 1.06, 0.96, 0.83, -0.17, -0.4, 0.3, -0.23, -0.45, 0.02,
    0.04, 0.48, 0.64, -2, -0.69, -0.13, -0.34, -0.05, 0.1, 0.27,
    0.55, 1.24, 0.29, -0.46, 0.62, -0.72, 1.62, -0.62, 0.38, 2.11,
    -1.03, -0.89, 1.27, -1.61, 1.12, 2.16, 0.43, 1.2, 1.03, 0.65,
    2.01, 1.09, 0.73, -1.4, -1.29, -0.97, -1.74, 0.83, -0.76, 2.16,
    -0.42, -0.01, 0.4, -0.37, -0.43, 0.24, 2.22, 0.31, -0.56, -1.49,
    0.42, -0.53, -0.35, 0.73, -0.56, -1.45, -1.05, -0.68, -0.61,
    1.89, -0.15, -0.64, -0.41, 0.34, -1.13, 0.07, 0.04, 0.92, 2.13,
    -0.6, 1.24, -0.08, 1.18, 2.19, 0.41, -0.74, -1.96, -1.95, -0.94,
    1.2, -0.62, -0.13, -0.02, -0.46, 1.47, 2.17, -3, -1.77, -0.36,
    0.37, -1.23, 0.47, 1.23, -0.59, 0.75, 0.61, -0.23, -0.7, 1.23,
    1.52, 1.35, -0.12, 0.93, 0.02, 0.83, -0.3, -0.06, 1.01, -1.38,
    -1, -0.02, 1.55, -0.76, 1.01, 0.27, -0.08, -0.9, 0.47, 1.39,
    0.03, -1.29, 1.16, -1.52, -2.52, -0.71, -0.29, -0.44, -0.34,
    -0.04, 0.89, -0.36, -0.73, 1.75, -0.09, 1.11, -0.02, -1.49, 2.7,
    -0.74, 1.07, -0.03, -0.54, -1.93, 1.01, -0.28, -1.25, 1.25, -0.19,
    0.91, -2.16, 1.29, -0.43, -0.26, 1.33, 0.84, -1.47, -0.62, 0.11,
    -1.84, -1.62, 0.23, 0.92, -0.33, 1.29, 0.06, -2.2, -0.52, 1.08,
    -0.24, -0.91, -0.87
};

std::vector<double> init {
    -0.5, 0.13, -0.08, 0.89, 0.12, 0.32, -0.58, 0.71, -0.83, -0.36, 
    0.09, 0.1, -0.2, 0.74, 0.12, -0.03, -0.39, 0.51, -0.91, 2.31
};

TEST_F(IrlbaTester, Reference) {
    irlba::Irlba irb;
    irlba::NormalSampler norm(50);

    Eigen::MatrixXd X(25, 20);
    ASSERT_EQ(X.rows() * X.cols(), store.size());
    auto sIt = store.begin();
    for (size_t i = 0; i < X.cols(); ++i) {
        for (size_t j = 0; j < X.rows(); ++j, ++sIt) {
            X(j, i) = *sIt;
        }
    }

    Eigen::VectorXd initV(init.size());
    std::copy(init.begin(), init.end(), initV.begin());

    irb.set_number(3).set_work(7).set_init(initV).run(X, false, false, norm, U, V, S);
    EXPECT_EQ(S.size(), 3);
    EXPECT_EQ(U.cols(), 3);
    EXPECT_EQ(V.cols(), 3);

    std::vector<double> refS { 8.561070799, 8.123755521, 7.204680224 };
    for (size_t i = 0; i < refS.size(); ++i) {
        EXPECT_FLOAT_EQ(refS[i], S[i]);
    }
}


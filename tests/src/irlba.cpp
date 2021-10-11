#include <gtest/gtest.h>

#include "compare.h"
#include "NormalSampler.h"

#include "irlba/irlba.hpp"
#include "irlba/utils.hpp"

#include "Eigen/Dense"
#include <random>

Eigen::MatrixXd create_random_matrix(size_t nr, size_t nc) {
    Eigen::MatrixXd A(nr, nc);
    NormalSampler norm(42); 
    for (size_t i = 0; i < nc; ++i) {
        for (size_t j = 0; j < nr; ++j) {
            A(j, i) = norm();
        }
    }
    return A;
}

TEST(IrlbaTest, Exact) {
    // For the test, the key is that rank + workspace > min(nr, nc), in which
    // case we can be pretty confident of getting a near-exact match of the
    // true SVD. Otherwise it's more approximate and the test is weaker.
    int rank = 5;
    auto A = create_random_matrix(20, 10);

    irlba::Irlba irb;
    auto res = irb.set_number(rank).run(A);

    Eigen::BDCSVD svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    expect_equal_vectors(res.D, svd.singularValues().head(rank), 1e-8);
    expect_equal_column_vectors(res.U, svd.matrixU().leftCols(rank), 1e-8);
    expect_equal_column_vectors(res.V, svd.matrixV().leftCols(rank), 1e-8);
}

class IrlbaTester : public ::testing::TestWithParam<std::tuple<int, int, int> > {
protected:
    template<class Param>
    void assemble(Param param) {
        nr = std::get<0>(param);
        nc = std::get<1>(param);
        rank = std::get<2>(param);
        A = create_random_matrix(nr, nc);
    }

    size_t nr, nc, rank;
    Eigen::MatrixXd A;
};

TEST_P(IrlbaTester, Basic) {
    assemble(GetParam());

    irlba::Irlba irb;
    auto res = irb.set_number(rank).run(A);

    ASSERT_EQ(res.V.cols(), rank);
    ASSERT_EQ(res.U.cols(), rank);
    ASSERT_EQ(res.D.size(), rank);

    // Gives us singular values that are around about right. Unfortunately,
    // the singular values don't converge enough for an exact comparison.
    Eigen::BDCSVD svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    expect_equal_vectors(res.D, svd.singularValues().head(rank), 1e-6);
}

TEST_P(IrlbaTester, CenterScale) {
    assemble(GetParam());

    NormalSampler norm(42); 
    Eigen::VectorXd center(A.cols());
    for (auto& c : center) { c = norm(); }
    Eigen::VectorXd scale(A.cols());
    for (auto& s : scale) { s = std::abs(norm() + 1); }

    irlba::Irlba irb;
    auto res = irb.run(A, center, scale);

    Eigen::MatrixXd copy = A;
    for (size_t i = 0; i < A.cols(); ++i) {
        for (size_t j = 0; j < A.rows(); ++j) {
            copy(j, i) -= center(i);
            copy(j, i) /= scale(i);
        }
    }

    auto res2 = irb.run(copy);

    expect_equal_vectors(res.D, res2.D);
    expect_equal_column_vectors(res.U, res2.U);
    expect_equal_column_vectors(res.V, res2.V);
}

TEST_P(IrlbaTester, CenterScaleAgain) {
    assemble(GetParam());

    irlba::Irlba irb;
    auto ref = irb.run(A);

    auto res2 = irb.run<true>(A);
    EXPECT_NE(ref.D, res2.D);
    for (size_t i = 0; i < res2.U.cols(); ++i) {
        EXPECT_TRUE(std::abs(res2.U.col(i).sum()) < 1e-8);
    }    

    auto res3 = irb.run<true, true>(A);
    EXPECT_NE(ref.D, res3.D);
    EXPECT_NE(res2.D, res3.D);
    for (size_t i = 0; i < res3.U.cols(); ++i) {
        EXPECT_TRUE(std::abs(res3.U.col(i).sum()) < 1e-8);
    }    

    auto res4 = irb.run<false, true>(A);
    EXPECT_NE(ref.D, res4.D);
    EXPECT_NE(res2.D, res4.D);
    EXPECT_NE(res3.D, res4.D);
}

TEST_P(IrlbaTester, Exact) {
    assemble(GetParam());

    irlba::Irlba irb;
    Eigen::MatrixXd small = A.leftCols(3);
    auto res = irb.set_number(2).run(small);
      
    Eigen::BDCSVD svd(small, Eigen::ComputeThinU | Eigen::ComputeThinV);
    EXPECT_EQ(svd.singularValues().head(2), res.D);
    EXPECT_EQ(svd.matrixU().leftCols(2), res.U);
    EXPECT_EQ(svd.matrixV().leftCols(2), res.V);
}

TEST_P(IrlbaTester, ExactCenterScale) {
    assemble(GetParam());

    Eigen::MatrixXd small = A.leftCols(3);

    NormalSampler norm(50);
    Eigen::VectorXd center(small.cols());
    for (auto& c : center) { c = norm(); }
    Eigen::VectorXd scale(small.cols());
    for (auto& s : scale) { s = std::abs(norm() + 1); }

    irlba::Irlba irb;
    auto res = irb.set_number(2).run(small, center, scale);

    Eigen::MatrixXd copy = small;
    for (size_t i = 0; i < small.cols(); ++i) {
        for (size_t j = 0; j < small.rows(); ++j) {
            copy(j, i) -= center(i);
            copy(j, i) /= scale(i);
        }
    }

    auto res2 = irb.set_number(2).run(copy);

    EXPECT_EQ(res.U, res2.U);
    EXPECT_EQ(res.V, res2.V);
    EXPECT_EQ(res.D, res2.D);
}

TEST_P(IrlbaTester, Fails) {
    assemble(GetParam());

    irlba::Irlba irb;

    // Requested number of SVs > smaller dimension of the matrix.
    try {
        irb.set_number(100).run(A);
    } catch (const std::exception& e) {
        std::string message(e.what());
        EXPECT_EQ(message.find("requested"), 0);
    }

    // Initialization vector is not of the right length.
    Eigen::VectorXd init(1);
    try {
        irb.set_number(5).run(A, irlba::null_rng(), &init);
    } catch (const std::exception& e) {
        std::string message(e.what());
        EXPECT_EQ(message.find("initialization"), 0);
    }
}

INSTANTIATE_TEST_SUITE_P(
    IrlbaTest,
    IrlbaTester,
    ::testing::Combine(
        ::testing::Values(20, 50, 100), // number of rows
        ::testing::Values(20, 50, 100), // number of columns
        ::testing::Values(2, 5, 10) // rank of interest
    )
);

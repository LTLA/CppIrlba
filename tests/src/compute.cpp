#include <gtest/gtest.h>

#include "compare.h"
#include "NormalSampler.h"

#include "irlba/compute.hpp"
#include "irlba/utils.hpp"

#include "Eigen/Dense"
#include <random>

TEST(IrlbaTest, CompareToExact) {
    // For the test, the key is that rank + workspace > min(nr, nc), in which
    // case we can be pretty confident of getting a near-exact match of the
    // true SVD. Otherwise it's more approximate and the test is weaker.
    int rank = 5;
    auto A = create_random_matrix(20, 10);

    irlba::Options opt;
    opt.exact_for_large_number = false;
    auto res = irlba::compute(A, 5, opt);

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

    irlba::Options opt;
    auto res = irlba::compute(A, rank, opt);
    ASSERT_EQ(res.V.cols(), rank);
    ASSERT_EQ(res.U.cols(), rank);
    ASSERT_EQ(res.D.size(), rank);

    // Gives us singular values that are around about right. Unfortunately, the
    // U and V values don't converge enough for a decent comparison; we'll
    // limit this to the first column, as this seems to be the most accurate. 
    Eigen::BDCSVD svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    expect_equal_vectors(res.D, svd.singularValues().head(rank), 1e-6);
    expect_equal_column_vectors(res.U.leftCols(1), svd.matrixU().leftCols(1), 1e-4);
    expect_equal_column_vectors(res.V.leftCols(1), svd.matrixV().leftCols(1), 1e-4);

    // Also gives the same results when the matrices are row-major.
    typedef Eigen::Matrix<double, -1, -1, Eigen::RowMajor> RowEigenXd;
    auto rmres = irlba::compute<RowEigenXd>(A, rank, opt);
    expect_equal_matrix(res.U, rmres.U);
    expect_equal_matrix(res.V, rmres.V);
    expect_equal_vectors(res.D, rmres.D);

    // Also works with some custom initialization.
    auto init = create_random_vector(A.cols(), 1239);
    opt.initial = &init;
    auto res2 = irlba::compute(A, rank, opt);
    expect_equal_vectors(res.D, res2.D, 1e-6);
}

std::vector<Eigen::MatrixXd> spawn_center_scale(const Eigen::MatrixXd& A) {
    Eigen::MatrixXd centered = A, scaled = A, both = A;

    for (Eigen::Index i = 0; i < A.cols(); ++i) {
        double mean = A.col(i).sum() / A.rows();
        double var1 = 0, var2 = 0;
        for (Eigen::Index j = 0; j < A.rows(); ++j) {
            double x = A(j, i);
            centered(j, i) -= mean;
            both(j, i) -= mean;
            var1 += (x - mean)*(x - mean);
            var2 += x * x;
        }

        both.col(i) /= std::sqrt(var1/(A.rows() - 1));
        scaled.col(i) /= std::sqrt(var2/(A.rows() - 1));
    }

    return std::vector<Eigen::MatrixXd>{ centered, scaled, both };
}


TEST_P(IrlbaTester, CenterScale) {
    assemble(GetParam());

    // Computing references.
    auto spawned = spawn_center_scale(A);
    const auto& centered = spawned[0];
    const auto& scaled = spawned[1];
    const auto& both = spawned[2];

    // Comparing to the observed calculation.
    irlba::Options opt;
    auto ref = irlba::compute(A, true, true, rank, opt);

    {
        auto res = ref;
        auto res2 = irlba::compute(both, rank, opt);
        expect_equal_vectors(res.D, res2.D);
        expect_equal_column_vectors<true>(res.U, res2.U);
        expect_equal_column_vectors(res.V, res2.V);
        for (Eigen::Index i = 0; i < res2.U.cols(); ++i) {
            EXPECT_TRUE(std::abs(res2.U.col(i).sum()) < 1e-8);
        }    
    }

    {
        auto res = irlba::compute(A, true, false, rank, opt);
        auto res2 = irlba::compute(centered, rank, opt);
        expect_equal_vectors(res.D, res2.D);
        expect_equal_column_vectors<true>(res.U, res2.U);
        expect_equal_column_vectors(res.V, res2.V);
        for (Eigen::Index i = 0; i < res2.U.cols(); ++i) {
            EXPECT_TRUE(std::abs(res2.U.col(i).sum()) < 1e-8);
        }    
        EXPECT_NE(ref.D, res2.D);
    }

    {
        auto res = irlba::compute(A, false, true, rank, opt);
        auto res2 = irlba::compute(scaled, rank, opt);
        expect_equal_vectors(res.D, res2.D);
        expect_equal_column_vectors(res.U, res2.U);
        expect_equal_column_vectors(res.V, res2.V);
        EXPECT_NE(ref.D, res2.D);
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

TEST(IrlbaTest, SmallExact) {
    Eigen::MatrixXd small = create_random_matrix(10, 3);

    auto res = irlba::compute(small, 2, irlba::Options());

    Eigen::BDCSVD svd(small, Eigen::ComputeThinU | Eigen::ComputeThinV);
    EXPECT_EQ(svd.singularValues().head(2), res.D);
    EXPECT_EQ(svd.matrixU().leftCols(2), res.U);
    EXPECT_EQ(svd.matrixV().leftCols(2), res.V);
}

TEST(IrlbaTest, LargeExact) {
    Eigen::MatrixXd mat = create_random_matrix(20, 50);

    irlba::Options opt;
    auto res = irlba::compute(mat, 15, opt);

    Eigen::BDCSVD svd(mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
    EXPECT_EQ(svd.singularValues().head(15), res.D);
    EXPECT_EQ(svd.matrixU().leftCols(15), res.U);
    EXPECT_EQ(svd.matrixV().leftCols(15), res.V);

    // Works with the maximum number.
    auto res2 = irlba::compute(mat, 20, opt);
    EXPECT_EQ(svd.singularValues(), res2.D);
    EXPECT_EQ(svd.matrixU(), res2.U);
    EXPECT_EQ(svd.matrixV(), res2.V);

    // Works past the maximum number.
    opt.cap_number = true;
    auto res3 = irlba::compute(mat, 50, opt);
    EXPECT_EQ(svd.singularValues(), res3.D);
    EXPECT_EQ(svd.matrixU(), res3.U);
    EXPECT_EQ(svd.matrixV(), res3.V);
}

TEST(IrlbaTest, SmallExactCenterScale) {
    Eigen::MatrixXd small = create_random_matrix(10, 3);

    auto spawned = spawn_center_scale(small);
    const auto& centered = spawned[0];
    const auto& scaled = spawned[1];
    const auto& both = spawned[2];

    // Comparing to the observed calculation.
    irlba::Options opt;
    auto ref = irlba::compute(small, true, true, 2, opt);

    {
        auto res = ref;
        auto res2 = irlba::compute(both, 2, opt);
        expect_equal_vectors(res.D, res2.D);
        expect_equal_column_vectors(res.U, res2.U);
        expect_equal_column_vectors(res.V, res2.V);
        for (Eigen::Index i = 0; i < res2.U.cols(); ++i) {
            EXPECT_TRUE(std::abs(res2.U.col(i).sum()) < 1e-8);
        }    
    }

    {
        auto res = irlba::compute(small, true, false, 2, opt);
        auto res2 = irlba::compute(centered, 2, opt);
        expect_equal_vectors(res.D, res2.D);
        expect_equal_column_vectors(res.U, res2.U);
        expect_equal_column_vectors(res.V, res2.V);
        for (Eigen::Index i = 0; i < res2.U.cols(); ++i) {
            EXPECT_TRUE(std::abs(res2.U.col(i).sum()) < 1e-8);
        }    
        EXPECT_NE(ref.D, res2.D);
    }

    {
        auto res = irlba::compute(small, false, true, 2, opt);
        auto res2 = irlba::compute(scaled, 2, opt);
        expect_equal_vectors(res.D, res2.D);
        expect_equal_column_vectors(res.U, res2.U);
        expect_equal_column_vectors(res.V, res2.V);
        EXPECT_NE(ref.D, res2.D);
    }
}

TEST(IrlbaTester, Fails) {
    Eigen::MatrixXd A = create_random_matrix(20, 10);

    // Requested number of SVs > smaller dimension of the matrix.
    try {
        irlba::compute(A, 100, irlba::Options());
    } catch (const std::exception& e) {
        std::string message(e.what());
        EXPECT_TRUE(message.find("cannot be greater than") != std::string::npos);
    }

    // Requested number of SVs > smaller dimension of the matrix.
    try {
        irlba::Options opt;
        opt.exact_for_large_number = false;
        irlba::compute(A, 10, opt);
    } catch (const std::exception& e) {
        std::string message(e.what());
        EXPECT_TRUE(message.find("must be less than") != std::string::npos);
    }

    // Initialization vector is not of the right length.
    Eigen::VectorXd init(1);
    try {
        irlba::Options opt;
        opt.initial = &init;
        irlba::compute(A, 5, opt);
    } catch (const std::exception& e) {
        std::string message(e.what());
        EXPECT_EQ(message.find("initialization"), 0);
    }
}



#include <gtest/gtest.h>

#include "compare.h"
#include "NormalSampler.h"

#include "irlba/irlba.hpp"
#include "irlba/utils.hpp"

#include "Eigen/Dense"
#include <random>

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

TEST(IrlbaComplexTest, Exact) {
    // For the test, the key is that rank + workspace > min(nr, nc), in which
    // case we can be pretty confident of getting a near-exact match of the
    // true SVD. Otherwise it's more approximate and the test is weaker.
    int rank = 5;
    auto A = create_random_complex_matrix(20, 10);

    irlba::Irlba<Eigen::MatrixXcd> irb;
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
    irlba::Irlba irb;
    irb.set_number(rank);
    auto ref = irb.run(A, true, true);
    {
        auto res = ref;
        auto res2 = irb.run(both);
        expect_equal_vectors(res.D, res2.D);
        expect_equal_column_vectors<true>(res.U, res2.U);
        expect_equal_column_vectors(res.V, res2.V);
        for (size_t i = 0; i < res2.U.cols(); ++i) {
            EXPECT_TRUE(std::abs(res2.U.col(i).sum()) < 1e-8);
        }    
    }

    {
        auto res = irb.run(A, true, false);
        auto res2 = irb.run(centered);
        expect_equal_vectors(res.D, res2.D);
        expect_equal_column_vectors<true>(res.U, res2.U);
        expect_equal_column_vectors(res.V, res2.V);
        for (size_t i = 0; i < res2.U.cols(); ++i) {
            EXPECT_TRUE(std::abs(res2.U.col(i).sum()) < 1e-8);
        }    
        EXPECT_NE(ref.D, res2.D);
    }

    {
        auto res = irb.run(A, false, true);
        auto res2 = irb.run(scaled);
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

    irlba::Irlba irb;
    auto res = irb.set_number(2).run(small);
      
    Eigen::BDCSVD svd(small, Eigen::ComputeThinU | Eigen::ComputeThinV);
    EXPECT_EQ(svd.singularValues().head(2), res.D);
    EXPECT_EQ(svd.matrixU().leftCols(2), res.U);
    EXPECT_EQ(svd.matrixV().leftCols(2), res.V);
}

TEST(IrlbaTest, SmallExactCenterScale) {
    Eigen::MatrixXd small = create_random_matrix(10, 3);

    auto spawned = spawn_center_scale(small);
    const auto& centered = spawned[0];
    const auto& scaled = spawned[1];
    const auto& both = spawned[2];

    // Comparing to the observed calculation.
    irlba::Irlba irb;
    irb.set_number(2);
    auto ref = irb.run(small, true, true);
    {
        auto res = ref;
        auto res2 = irb.run(both);
        expect_equal_vectors(res.D, res2.D);
        expect_equal_column_vectors(res.U, res2.U);
        expect_equal_column_vectors(res.V, res2.V);
        for (size_t i = 0; i < res2.U.cols(); ++i) {
            EXPECT_TRUE(std::abs(res2.U.col(i).sum()) < 1e-8);
        }    
    }

    {
        auto res = irb.run(small, true, false);
        auto res2 = irb.run(centered);
        expect_equal_vectors(res.D, res2.D);
        expect_equal_column_vectors(res.U, res2.U);
        expect_equal_column_vectors(res.V, res2.V);
        for (size_t i = 0; i < res2.U.cols(); ++i) {
            EXPECT_TRUE(std::abs(res2.U.col(i).sum()) < 1e-8);
        }    
        EXPECT_NE(ref.D, res2.D);
    }

    {
        auto res = irb.run(small, false, true);
        auto res2 = irb.run(scaled);
        expect_equal_vectors(res.D, res2.D);
        expect_equal_column_vectors(res.U, res2.U);
        expect_equal_column_vectors(res.V, res2.V);
        EXPECT_NE(ref.D, res2.D);
    }
}

TEST(IrlbaTester, Fails) {
    Eigen::MatrixXd A = create_random_matrix(20, 10);

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



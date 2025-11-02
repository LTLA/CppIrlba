#include <gtest/gtest.h>

#include "compare.h"
#include "NormalSampler.h"
#include "sparse.h"

#include "irlba/pca.hpp"
#include "irlba/utils.hpp"

#include "Eigen/Dense"
#include <random>

class PcaTest : public ::testing::TestWithParam<std::tuple<int, int, int> > {
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

static std::vector<Eigen::MatrixXd> spawn_center_scale(const Eigen::MatrixXd& A) {
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

TEST_P(PcaTest, CenterScale) {
    assemble(GetParam());

    // Computing references.
    auto spawned = spawn_center_scale(A);
    const auto& centered = spawned[0];
    const auto& scaled = spawned[1];
    const auto& both = spawned[2];

    // Comparing to the observed calculation.
    irlba::Options opt;
    auto ref = irlba::pca(A, true, true, rank, opt);

    {
        const auto& res = ref;
        irlba::SimpleMatrix<Eigen::VectorXd, Eigen::MatrixXd, decltype(&both)> wrapper(&both);
        auto res2 = irlba::compute(wrapper, rank, opt);

        expect_equal_vectors(res.D, res2.D);
        expect_equal_column_vectors<true>(res.U, res2.U);
        expect_equal_column_vectors(res.V, res2.V);
        for (Eigen::Index i = 0; i < res2.U.cols(); ++i) {
            EXPECT_TRUE(std::abs(res2.U.col(i).sum()) < 1e-8);
        }    
    }

    {
        auto res = irlba::pca(A, true, false, rank, opt);
        irlba::SimpleMatrix<Eigen::VectorXd, Eigen::MatrixXd, decltype(&centered)> wrapper(&centered);
        auto res2 = irlba::compute(wrapper, rank, opt);

        expect_equal_vectors(res.D, res2.D);
        expect_equal_column_vectors<true>(res.U, res2.U);
        expect_equal_column_vectors(res.V, res2.V);
        for (Eigen::Index i = 0; i < res2.U.cols(); ++i) {
            EXPECT_TRUE(std::abs(res2.U.col(i).sum()) < 1e-8);
        }    
        EXPECT_NE(ref.D, res2.D);
    }

    {
        auto res = irlba::pca(A, false, true, rank, opt);
        irlba::SimpleMatrix<Eigen::VectorXd, Eigen::MatrixXd, decltype(&scaled)> wrapper(&scaled);
        auto res2 = irlba::compute(wrapper, rank, opt);

        expect_equal_vectors(res.D, res2.D);
        expect_equal_column_vectors(res.U, res2.U);
        expect_equal_column_vectors(res.V, res2.V);
        EXPECT_NE(ref.D, res2.D);
    }
}

INSTANTIATE_TEST_SUITE_P(
    Pca,
    PcaTest,
    ::testing::Combine(
        ::testing::Values(20, 50, 100), // number of rows
        ::testing::Values(20, 50, 100), // number of columns
        ::testing::Values(2, 5, 10) // rank of interest
    )
);

TEST(Pca, SmallExactCenterScale) {
    Eigen::MatrixXd small = create_random_matrix(10, 3);

    auto spawned = spawn_center_scale(small);
    const auto& centered = spawned[0];
    const auto& scaled = spawned[1];
    const auto& both = spawned[2];

    // Comparing to the observed calculation.
    irlba::Options opt;
    auto ref = irlba::pca(small, true, true, 2, opt);

    {
        const auto& res = ref;
        irlba::SimpleMatrix<Eigen::VectorXd, Eigen::MatrixXd, decltype(&both)> wrapper(&both);
        auto res2 = irlba::compute(wrapper, 2, opt);

        expect_equal_vectors(res.D, res2.D);
        expect_equal_column_vectors(res.U, res2.U);
        expect_equal_column_vectors(res.V, res2.V);
        for (Eigen::Index i = 0; i < res2.U.cols(); ++i) {
            EXPECT_TRUE(std::abs(res2.U.col(i).sum()) < 1e-8);
        }    
    }

    {
        auto res = irlba::pca(small, true, false, 2, opt);
        irlba::SimpleMatrix<Eigen::VectorXd, Eigen::MatrixXd, decltype(&centered)> wrapper(&centered);
        auto res2 = irlba::compute(wrapper, 2, opt);

        expect_equal_vectors(res.D, res2.D);
        expect_equal_column_vectors(res.U, res2.U);
        expect_equal_column_vectors(res.V, res2.V);
        for (Eigen::Index i = 0; i < res2.U.cols(); ++i) {
            EXPECT_TRUE(std::abs(res2.U.col(i).sum()) < 1e-8);
        }    
        EXPECT_NE(ref.D, res2.D);
    }

    {
        auto res = irlba::pca(small, false, true, 2, opt);
        irlba::SimpleMatrix<Eigen::VectorXd, Eigen::MatrixXd, decltype(&scaled)> wrapper(&scaled);
        auto res2 = irlba::compute(wrapper, 2, opt);

        expect_equal_vectors(res.D, res2.D);
        expect_equal_column_vectors(res.U, res2.U);
        expect_equal_column_vectors(res.V, res2.V);
        EXPECT_NE(ref.D, res2.D);
    }
}

TEST(Pca, Sparse) {
    auto simulated = simulate_compressed_sparse(76, 128);
    auto A = create_dense_matrix(simulated);
    auto B = create_sparse_matrix(simulated);

    irlba::Options opt;
    opt.extra_work = 7;
    auto res = irlba::pca(A, true, true, 8, opt);
    auto res2 = irlba::pca(B, true, true, 8, opt);

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

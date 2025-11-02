#include <gtest/gtest.h>

#include "compare.h"
#include "NormalSampler.h"
#include "sparse.h"

#include "irlba/compute.hpp"
#include "irlba/Matrix/simple.hpp"
#include "irlba/Matrix/sparse.hpp"
#include "irlba/utils.hpp"

#include "Eigen/Dense"
#include <random>

TEST(Compute, CompareToExact) {
    // For the test, the key is that rank + workspace > min(nr, nc), in which
    // case we can be pretty confident of getting a near-exact match of the
    // true SVD. Otherwise it's more approximate and the test is weaker.
    int rank = 5;
    auto A = create_random_matrix(20, 10);

    irlba::Options opt;
    opt.exact_for_large_number = false;
    irlba::SimpleMatrix<Eigen::VectorXd, Eigen::MatrixXd, decltype(&A)> wrapped(&A);
    auto res = irlba::compute(wrapped, 5, opt);

    Eigen::JacobiSVD<decltype(A), Eigen::ComputeThinU | Eigen::ComputeThinV> svd(A);
    expect_equal_vectors(res.D, svd.singularValues().head(rank), 1e-8);
    expect_equal_column_vectors(res.U, svd.matrixU().leftCols(rank), 1e-8);
    expect_equal_column_vectors(res.V, svd.matrixV().leftCols(rank), 1e-8);
}

class ComputeTest : public ::testing::TestWithParam<std::tuple<int, int, int> > {
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

TEST_P(ComputeTest, Basic) {
    assemble(GetParam());
    irlba::SimpleMatrix<Eigen::VectorXd, Eigen::MatrixXd, decltype(&A)> wrapped(&A);

    irlba::Options opt;
    opt.convergence_tolerance = 1e-9;
    auto res = irlba::compute(wrapped, rank, opt);
    ASSERT_EQ(res.V.cols(), rank);
    ASSERT_EQ(res.U.cols(), rank);
    ASSERT_EQ(res.D.size(), rank);

    // Gives us singular values that are around about right.
    Eigen::JacobiSVD<decltype(A), Eigen::ComputeThinU | Eigen::ComputeThinV> svd(A);
    expect_equal_vectors(res.D, svd.singularValues().head(rank), 1e-6);
    expect_equal_column_vectors(res.U, svd.matrixU().leftCols(rank), 1e-6);
    expect_equal_column_vectors(res.V, svd.matrixV().leftCols(rank), 1e-6);

    // Also gives the same results when the matrices are row-major.
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> Arow(A);
    irlba::SimpleMatrix<Eigen::VectorXd, Eigen::MatrixXd, decltype(&Arow)> wrapped_row(&Arow);
    auto rmres = irlba::compute(wrapped_row, rank, opt);
    expect_equal_matrix(res.U, rmres.U);
    expect_equal_matrix(res.V, rmres.V);
    expect_equal_vectors(res.D, rmres.D);

    // Also works with some custom initialization.
    auto init = create_random_vector(A.cols(), 1239);
    opt.initial = &init;
    auto res2 = irlba::compute(wrapped, rank, opt);
    expect_equal_vectors(res.D, res2.D, 1e-6);
}

INSTANTIATE_TEST_SUITE_P(
    Compute,
    ComputeTest,
    ::testing::Combine(
        ::testing::Values(20, 50, 100), // number of rows
        ::testing::Values(20, 50, 100), // number of columns
        ::testing::Values(2, 5, 10) // rank of interest
    )
);

TEST(IrlbaTest, SmallExact) {
    Eigen::MatrixXd small = create_random_matrix(10, 3);
    irlba::SimpleMatrix<Eigen::VectorXd, Eigen::MatrixXd, decltype(&small)> wrapped(&small);
    auto res = irlba::compute(wrapped, 2, irlba::Options());

    Eigen::JacobiSVD<decltype(small), Eigen::ComputeThinU | Eigen::ComputeThinV> svd(small);
    EXPECT_EQ(svd.singularValues().head(2), res.D);
    EXPECT_EQ(svd.matrixU().leftCols(2), res.U);
    EXPECT_EQ(svd.matrixV().leftCols(2), res.V);
}

TEST(Compute, LargeExact) {
    Eigen::MatrixXd mat = create_random_matrix(20, 50);
    irlba::SimpleMatrix<Eigen::VectorXd, Eigen::MatrixXd, decltype(&mat)> wrapped(&mat);
    irlba::Options opt;
    auto res = irlba::compute(wrapped, 15, opt);

    Eigen::JacobiSVD<decltype(mat), Eigen::ComputeThinU | Eigen::ComputeThinV> svd(mat);
    EXPECT_EQ(svd.singularValues().head(15), res.D);
    EXPECT_EQ(svd.matrixU().leftCols(15), res.U);
    EXPECT_EQ(svd.matrixV().leftCols(15), res.V);

    // Works with the maximum number.
    auto res2 = irlba::compute(wrapped, 20, opt);
    EXPECT_EQ(svd.singularValues(), res2.D);
    EXPECT_EQ(svd.matrixU(), res2.U);
    EXPECT_EQ(svd.matrixV(), res2.V);

    // Works past the maximum number.
    opt.cap_number = true;
    auto res3 = irlba::compute(wrapped, 50, opt);
    EXPECT_EQ(svd.singularValues(), res3.D);
    EXPECT_EQ(svd.matrixU(), res3.U);
    EXPECT_EQ(svd.matrixV(), res3.V);
}

TEST(Compute, Fails) {
    Eigen::MatrixXd A = create_random_matrix(20, 10);
    irlba::SimpleMatrix<Eigen::VectorXd, Eigen::MatrixXd, decltype(&A)> wrapped(&A);

    // Requested number of SVs > smaller dimension of the matrix.
    std::string message;
    try {
        irlba::compute(wrapped, 100, irlba::Options());
    } catch (const std::exception& e) {
        message = e.what();
    }
    EXPECT_TRUE(message.find("cannot be greater than") != std::string::npos);

    // Requested number of SVs > smaller dimension of the matrix.
    message.clear();
    try {
        irlba::Options opt;
        opt.exact_for_large_number = false;
        irlba::compute(wrapped, 10, opt);
    } catch (const std::exception& e) {
        message = e.what();
    }
    EXPECT_TRUE(message.find("must be less than") != std::string::npos);

    // Initialization vector is not of the right length.
    message.clear();
    Eigen::VectorXd init(1);
    try {
        irlba::Options opt;
        opt.initial = &init;
        irlba::compute(wrapped, 2, opt);
    } catch (const std::exception& e) {
        message = e.what();
    }
    EXPECT_EQ(message.find("initialization"), 0);
}

TEST(Compute, Sparse) {
    auto simulated = simulate_compressed_sparse(99, 64);
    auto A = create_dense_matrix(simulated);
    auto B = create_sparse_matrix(simulated);

    irlba::Options opt;
    opt.extra_work = 7;

    irlba::SimpleMatrix<Eigen::VectorXd, Eigen::MatrixXd, decltype(&A)> wrapped(&A);
    auto res = irlba::compute(wrapped, 8, opt);
    irlba::SimpleMatrix<Eigen::VectorXd, Eigen::MatrixXd, decltype(&B)> wrapped2(&B);
    auto res2 = irlba::compute(wrapped2, 8, opt);

    expect_equal_vectors(res.D, res2.D);
    expect_equal_column_vectors(res.U, res2.U);
    expect_equal_column_vectors(res.V, res2.V);

    // Checking our custom sparse matrix.
    {
        irlba::ParallelSparseMatrix<
            Eigen::VectorXd,
            Eigen::MatrixXd,
            decltype(simulated.values),
            decltype(simulated.indices),
            decltype(simulated.nzeros)
        > psparse(simulated.rows, simulated.cols, simulated.values, simulated.indices, simulated.nzeros, true, 1);
        auto res3 = irlba::compute(psparse, 8, opt);

        expect_equal_vectors(res.D, res3.D);
        expect_equal_column_vectors(res.U, res3.U);
        expect_equal_column_vectors(res.V, res3.V);
    }

    // Comparing it to the reference. 
    {
        irlba::Options opt;
        opt.extra_work = 20;
        auto res = irlba::compute(wrapped2, 13, opt);

        // Bumping up the tolerance as later SV's tend to be a bit more variable.
        Eigen::JacobiSVD<decltype(A), Eigen::ComputeThinU | Eigen::ComputeThinV> ref(A);
        expect_equal_vectors(res.D, ref.singularValues().head(13), 1e-5);
        expect_equal_column_vectors(res.U, ref.matrixU().leftCols(13), 1e-5);
        expect_equal_column_vectors(res.V, ref.matrixV().leftCols(13), 1e-5);
    }
}

#include <gtest/gtest.h>

#ifdef TEST_CUSTOM_PARALLEL
// This must be done before including irlba.
#include "../custom_parallel.h"
#endif

#include "irlba/Matrix/sparse.hpp"
#include "Eigen/Dense"

#include "../compare.h"
#include "../sparse.h"
#include "../NormalSampler.h"

#include <random>
#include <vector>
#include <cstddef>

class ParallelSparseMatrixTest : public ::testing::TestWithParam<std::tuple<int, int, int> > {
protected:
    void SetUp() {
        auto param = GetParam();
        nr = std::get<0>(param);
        nc = std::get<1>(param);
        nt = std::get<2>(param);

        auto data = simulate_compressed_sparse(nr, nc);
        control = create_dense_matrix(data);

        values.swap(data.values);
        indices.swap(data.indices);
        nzeros.swap(data.nzeros);
    }

    std::size_t nr, nc;
    int nt;
    Eigen::MatrixXd control;

    std::vector<double> values;
    std::vector<int> indices;
    std::vector<std::size_t> nzeros;
};

TEST_P(ParallelSparseMatrixTest, Basic) {
    irlba::EigenThreadScope tscope(nt);

    irlba::ParallelSparseMatrix<Eigen::VectorXd, Eigen::MatrixXd, decltype(values), decltype(indices), decltype(nzeros)> A(nr, nc, values, indices, nzeros, true, nt);
    EXPECT_EQ(A.rows(), nr);
    EXPECT_EQ(A.cols(), nc);

    // Testing all getters.
    EXPECT_EQ(A.get_indices().size(), indices.size());
    EXPECT_EQ(A.get_values().size(), values.size());
    EXPECT_EQ(A.get_pointers().size(), nc + 1);
    if (nt > 1) {
        EXPECT_EQ(A.get_primary_starts().size(), nt);
        EXPECT_EQ(A.get_primary_ends().size(), nt);
        EXPECT_EQ(A.get_secondary_nonzero_starts().size(), nt + 1);
        EXPECT_EQ(A.get_secondary_nonzero_starts().front().size(), nc);
    }

    irlba::ParallelSparseMatrix<Eigen::VectorXd, Eigen::MatrixXd, decltype(values), decltype(indices), decltype(nzeros)> A2(nc, nr, values, indices, nzeros, false, nt);
    EXPECT_EQ(A2.rows(), nc);
    EXPECT_EQ(A2.cols(), nr);

    // Realizes correctly.
    Eigen::MatrixXd realized, realized2;
    auto realizer = A.new_realize_workspace();
    realizer->realize_copy(realized);
    expect_equal_matrix(control, realized);

    auto realizer2 = A2.new_realize_workspace();
    realizer2->realize_copy(realized2);
    realized2.adjointInPlace();
    expect_equal_matrix(control, realized2);

    // Vector multiplies correctly.
    {
        auto vec = create_random_vector(nc, nr * nc * nt / 10);
        Eigen::VectorXd ref = control * vec;

        Eigen::VectorXd obs(nr);
        auto work = A.new_workspace();
        work->multiply(vec, obs);
        expect_equal_vectors(ref, obs, 0);

        Eigen::VectorXd obs2(nr);
        auto awork2 = A2.new_adjoint_workspace();
        awork2->multiply(vec, obs2);
        expect_equal_vectors(ref, obs2);

        // Check for correct re-zeroing of the provided buffers.
        work->multiply(vec, obs);
        expect_equal_vectors(ref, obs, 0);
        awork2->multiply(vec, obs2);
        expect_equal_vectors(ref, obs2);
    }

    // Adjoint multiplies correctly.
    {
        auto vec = create_random_vector(nr, nr * nc * nt / 20);
        Eigen::VectorXd ref = control.adjoint() * vec;

        Eigen::VectorXd obs(nc);
        auto awork = A.new_adjoint_workspace();
        awork->multiply(vec, obs);
        expect_equal_vectors(ref, obs);

        Eigen::VectorXd obs2(nc);
        auto work2 = A2.new_workspace();
        work2->multiply(vec, obs2);
        expect_equal_vectors(ref, obs2);

        // Check for correct re-zeroing of the provided buffers.
        awork->multiply(vec, obs);
        expect_equal_vectors(ref, obs);
        work2->multiply(vec, obs2);
        expect_equal_vectors(ref, obs2);
    }
}

INSTANTIATE_TEST_SUITE_P(
    ParallelSparseMatrix,
    ParallelSparseMatrixTest,
    ::testing::Combine(
        ::testing::Values(10, 49, 97), // number of rows
        ::testing::Values(10, 51, 88), // number of columns
        ::testing::Values(1, 3) // number of threads
    )
);

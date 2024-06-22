#include <gtest/gtest.h>

#ifdef TEST_CUSTOM_PARALLEL
// This must be done before including irlba.
#include "custom_parallel.h"
#endif

#include "irlba/parallel.hpp"
#include "irlba/irlba.hpp"
#include "Eigen/Dense"
#include <random>

#include "compare.h"
#include "NormalSampler.h"

class ParallelSparseMatrixTestCore {
protected:
    template<class Param>
    void assemble(const Param& param) {
        nr = std::get<0>(param);
        nc = std::get<1>(param);
        nt = std::get<2>(param);

        control = Eigen::MatrixXd(nr, nc);
        control.setZero();

        std::mt19937_64 rng(nr * nc);
        std::uniform_real_distribution udist(0.0, 1.0);
        std::normal_distribution ndist;

        nzeros.resize(nc + 1);

        for (size_t c = 0; c < nc; ++c) {
            for (size_t r = 0; r < nr; ++r) {
                if (udist(rng) < 0.2) {
                    double val = ndist(rng);
                    control(r, c) = val;
                    values.push_back(val);
                    indices.push_back(r);
                    ++(nzeros[c+1]);
                }
            }
        }

        for (size_t c = 0; c < nc; ++c) {
            nzeros[c+1] += nzeros[c];
        }
    }

    size_t nr, nc;
    int nt;
    Eigen::MatrixXd control;

    std::vector<double> values;
    std::vector<int> indices;
    std::vector<size_t> nzeros;
};

class ParallelSparseMatrixTest : public ::testing::TestWithParam<std::tuple<int, int, int> >, public ParallelSparseMatrixTestCore {};

TEST_P(ParallelSparseMatrixTest, Basic) {
    auto param = GetParam();
    assemble(param);
    irlba::EigenThreadScope tscope(nt);

    irlba::ParallelSparseMatrix A(nr, nc, values, indices, nzeros, true, nt);
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

    irlba::ParallelSparseMatrix A2(nc, nr, values, indices, nzeros, false, nt);
    EXPECT_EQ(A2.rows(), nc);
    EXPECT_EQ(A2.cols(), nr);

    // Realizes correctly.
    auto realized = A.template realize<Eigen::MatrixXd>();
    auto realized2 = A2.template realize<Eigen::MatrixXd>();
    realized2.adjointInPlace();
    bool okay1 = true, okay2 = true;

    for (size_t c = 0; c < nc; ++c) {
        auto col = realized.col(c);
        auto col2 = realized2.col(c);
        auto ref = control.col(c);

        for (size_t r = 0; r < nr; ++r) {
            if (col[r] != ref[r]) {
                okay1 = false;
                break;
            } else if (col2[r] != ref[r]) {
                okay2 = false;
                break;
            }
        }
    }

    EXPECT_TRUE(okay1);
    EXPECT_TRUE(okay2);

    // Vector multiplies correctly.
    {
        auto vec = create_random_vector(nc, nr * nc * nt / 10);
        Eigen::VectorXd ref = control * vec;

        Eigen::VectorXd obs(nr);
        auto work = A.workspace();
        A.multiply(vec, work, obs);
        expect_equal_vectors(ref, obs, 0);

        Eigen::VectorXd obs2(nr);
        auto awork2 = A2.adjoint_workspace();
        A2.adjoint_multiply(vec, awork2, obs2);
        expect_equal_vectors(ref, obs2);

        // Works on expressions.
        {
            auto expr = vec * 2;
            ref = control * expr;
            A.multiply(expr, work, obs);
            expect_equal_vectors(ref, obs, 0);
        }
    }

    // Adjoint multiplies correctly.
    {
        auto vec = create_random_vector(nr, nr * nc * nt / 20);
        Eigen::VectorXd ref = control.adjoint() * vec;

        Eigen::VectorXd obs(nc);
        auto awork = A.adjoint_workspace();
        A.adjoint_multiply(vec, awork, obs);
        expect_equal_vectors(ref, obs);

        Eigen::VectorXd obs2(nc);
        auto work2 = A2.workspace();
        A2.multiply(vec, work2, obs2);
        expect_equal_vectors(ref, obs2);

        // Works on expressions.
        {
            auto expr = vec * 2;
            ref = control.adjoint() * expr;
            A.adjoint_multiply(expr, awork, obs);
            expect_equal_vectors(ref, obs);
        }
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

class ParallelSparseMatrixIrlbaTest : public ::testing::TestWithParam<std::tuple<int, int, int> >, public ParallelSparseMatrixTestCore {};

TEST_P(ParallelSparseMatrixIrlbaTest, Basic) {
    auto param = GetParam();
    assemble(param);
    irlba::ParallelSparseMatrix A(nr, nc, values, indices, nzeros, true, nt);
    irlba::EigenThreadScope tscope(nt);
    irlba::Options opt;

    // Raw.
    {
        auto ref = irlba::compute(control, 5, opt);
        auto obs = irlba::compute(A, 5, opt);
        expect_equal_vectors(ref.D, obs.D);
        expect_equal_column_vectors(ref.U, obs.U);
        expect_equal_column_vectors(ref.V, obs.V);
    }

    std::uniform_real_distribution udist;
    std::mt19937_64 rng(nr * nc * 13);
    Eigen::VectorXd rando(nc);
    for (auto& r : rando) {
        r = udist(rng);
    }

    // Centered.
    {
        auto ref = irlba::compute(irlba::Centered(control, rando), 5, opt);
        auto obs = irlba::compute(irlba::Centered(A, rando), 5, opt);
        expect_equal_vectors(ref.D, obs.D);
        expect_equal_column_vectors(ref.U, obs.U);
        expect_equal_column_vectors(ref.V, obs.V);
    }

    // Scaled.
    {
        auto ref = irlba::compute(irlba::make_Scaled<true>(control, rando, false), 5, opt);
        auto obs = irlba::compute(irlba::make_Scaled<true>(A, rando, false), 5, opt);
        expect_equal_vectors(ref.D, obs.D);
        expect_equal_column_vectors(ref.U, obs.U);
        expect_equal_column_vectors(ref.V, obs.V);
    }
}

INSTANTIATE_TEST_SUITE_P(
    ParallelSparseMatrix,
    ParallelSparseMatrixIrlbaTest,
    ::testing::Combine(
        ::testing::Values(25, 51), // number of rows
        ::testing::Values(32, 47), // number of columns
        ::testing::Values(1, 3) // number of threads
    )
);

#include <gtest/gtest.h>

#include "irlba/parallel.hpp"
#include "irlba/irlba.hpp"
#include "Eigen/Dense"
#include <random>
#include "compare.h"

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

    irlba::ParallelSparseMatrix A(nr, nc, values, indices, nzeros, nt);
    EXPECT_EQ(A.rows(), nr);
    EXPECT_EQ(A.cols(), nc);

    irlba::ParallelSparseMatrix<false> A2(nc, nr, values, indices, nzeros, nt);
    EXPECT_EQ(A2.rows(), nc);
    EXPECT_EQ(A2.cols(), nr);

    // Realizes correctly.
    auto realized = A.realize();
    auto realized2 = A2.realize();
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

    std::normal_distribution ndist;
    std::mt19937_64 rng(nr * nc * 13);

    // Vector multiplies correctly.
    {
        Eigen::VectorXd vec(nc);
        for (auto& v : vec) {
            v = ndist(rng);
        }

        Eigen::VectorXd ref = control * vec;

        Eigen::VectorXd obs(nr);
        A.multiply(vec, obs);
        expect_equal_vectors(ref, obs, 0);

        Eigen::VectorXd obs2(nr);
        A2.adjoint_multiply(vec, obs2);
        expect_equal_vectors(ref, obs2);
    }

    // Adjoint multiplies correctly.
    {
        Eigen::VectorXd vec(nr);
        for (auto& v : vec) {
            v = ndist(rng);
        }

        Eigen::VectorXd ref = control.adjoint() * vec;

        Eigen::VectorXd obs(nc);
        A.adjoint_multiply(vec, obs);
        expect_equal_vectors(ref, obs);

        Eigen::VectorXd obs2(nc);
        A2.multiply(vec, obs2);
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

class ParallelSparseMatrixIrlbaTest : public ::testing::TestWithParam<std::tuple<int, int, int> >, public ParallelSparseMatrixTestCore {};

TEST_P(ParallelSparseMatrixIrlbaTest, Basic) {
    auto param = GetParam();
    assemble(param);
    irlba::ParallelSparseMatrix A(nr, nc, values, indices, nzeros, nt);
    irlba::Irlba irb;

    // Raw.
    {
        auto ref = irb.run(control);
        auto obs = irb.run(A);
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
        auto ref = irb.run(irlba::Centered(&control, &rando));
        auto obs = irb.run(irlba::Centered(&A, &rando));
        expect_equal_vectors(ref.D, obs.D);
        expect_equal_column_vectors(ref.U, obs.U);
        expect_equal_column_vectors(ref.V, obs.V);
    }

    // Scaled.
    {
        auto ref = irb.run(irlba::Scaled(&control, &rando));
        auto obs = irb.run(irlba::Scaled(&A, &rando));
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



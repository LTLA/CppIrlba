#include <gtest/gtest.h>

#include "irlba/Matrix/simple.hpp"

#include "../sparse.h"
#include "../NormalSampler.h"
#include "../compare.h"

#include "Eigen/Dense"

class SimpleTest : public ::testing::Test {
protected:
    inline static Eigen::MatrixXd A;

    static void SetUpTestSuite() {
        A = create_random_matrix(30, 19);
    }
};

TEST_F(SimpleTest, Basic) {
    irlba::SimpleMatrix<Eigen::VectorXd, Eigen::MatrixXd, decltype(&A)> wrapped(&A);
    EXPECT_EQ(wrapped.rows(), A.rows());
    EXPECT_EQ(wrapped.cols(), A.cols());

    auto realizer = wrapped.new_realize_workspace();
    Eigen::MatrixXd realized_buffer;
    const auto& realized = realizer->realize(realized_buffer);
    EXPECT_EQ(&realized, &A);

    realizer->realize_copy(realized_buffer);
    expect_equal_matrix(A, realized_buffer);
}

TEST_F(SimpleTest, Multiply) {
    irlba::SimpleMatrix<Eigen::VectorXd, Eigen::MatrixXd, decltype(&A)> wrapped(&A);

    auto C = create_random_vector(19, 5555);
    Eigen::VectorXd expected = A * C;

    auto wrk = wrapped.new_workspace();
    Eigen::VectorXd output(30);
    wrk->multiply(C, output);
    expect_equal_matrix(expected, output);
}

TEST_F(SimpleTest, AdjointMultiply) {
    irlba::SimpleMatrix<Eigen::VectorXd, Eigen::MatrixXd, decltype(&A)> wrapped(&A);

    auto C = create_random_vector(30, 6666);
    Eigen::VectorXd expected = A.adjoint() * C;

    auto wrk = wrapped.new_adjoint_workspace();
    Eigen::VectorXd output(19);
    wrk->multiply(C, output);
    expect_equal_matrix(expected, output);
}

TEST(Simple, Sparse) {
    auto simulated = simulate_compressed_sparse(39, 69);
    auto B = create_sparse_matrix(simulated);

    irlba::SimpleMatrix<Eigen::VectorXd, Eigen::MatrixXd, decltype(&B)> wrapped(&B);
    EXPECT_EQ(wrapped.rows(), B.rows());
    EXPECT_EQ(wrapped.cols(), B.cols());

    {
        auto realizer = wrapped.new_realize_workspace();
        Eigen::MatrixXd realized_buffer;
        const auto& realized = realizer->realize(realized_buffer);
        EXPECT_EQ(&realized, &realized_buffer);
        expect_equal_matrix(Eigen::MatrixXd(B), realized);
    }

    {
        auto C = create_random_vector(69, 222);
        Eigen::VectorXd expected = B * C;
        auto wrk = wrapped.new_workspace();
        Eigen::VectorXd output(39);
        wrk->multiply(C, output);
        expect_equal_matrix(expected, output);
    }

    {
        auto C = create_random_vector(39, 111);
        Eigen::VectorXd expected = B.adjoint() * C;
        auto wrk = wrapped.new_adjoint_workspace();
        Eigen::VectorXd output(69);
        wrk->multiply(C, output);
        expect_equal_matrix(expected, output);
    }
}

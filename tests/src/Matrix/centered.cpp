#include <gtest/gtest.h>

#include "irlba/Matrix/centered.hpp"
#include "irlba/Matrix/simple.hpp"

#include "../NormalSampler.h"
#include "../compare.h"

#include "Eigen/Dense"

class CenteredTest : public ::testing::Test {
protected:
    inline static Eigen::MatrixXd A;
    inline static Eigen::VectorXd B;

    static void SetUpTestSuite() {
        A = create_random_matrix(20, 10);
        B = create_random_vector(10);
    }
};

TEST_F(CenteredTest, Basic) {
    irlba::SimpleMatrix<Eigen::VectorXd, Eigen::MatrixXd, decltype(&A)> wrapped(&A);
    irlba::CenteredMatrix<Eigen::VectorXd, Eigen::MatrixXd> centered(&wrapped, &B);
    EXPECT_EQ(centered.rows(), A.rows());
    EXPECT_EQ(centered.cols(), A.cols());

    Eigen::MatrixXd expected = A;
    for (Eigen::Index c = 0; c < A.cols(); ++c) {
        for (Eigen::Index r = 0; r < A.rows(); ++r) {
            expected(r, c) -= B[c];
        }
    }

    auto realizer = centered.new_realize_workspace();
    Eigen::MatrixXd realized;
    realizer->realize_copy(realized);
    expect_equal_matrix(expected, realized);
}

TEST_F(CenteredTest, Multiply) {
    irlba::SimpleMatrix<Eigen::VectorXd, Eigen::MatrixXd, decltype(&A)> wrapped(&A);
    irlba::CenteredMatrix<Eigen::VectorXd, Eigen::MatrixXd> centered(&wrapped, &B);
    auto realizer = centered.new_realize_workspace();
    Eigen::MatrixXd realized;
    realizer->realize_copy(realized);

    auto C = create_random_vector(10, 1234);
    Eigen::VectorXd expected = realized * C;

    auto wrk = centered.new_workspace();
    Eigen::VectorXd output(20);
    wrk->multiply(C, output);
    expect_equal_matrix(expected, output);

    // Check that non-zeroed outputs have no effect.
    wrk->multiply(C, output);
    expect_equal_matrix(expected, output);
}

TEST_F(CenteredTest, AdjointMultiply) {
    irlba::SimpleMatrix<Eigen::VectorXd, Eigen::MatrixXd, decltype(&A)> wrapped(&A);
    irlba::CenteredMatrix<Eigen::VectorXd, Eigen::MatrixXd> centered(&wrapped, &B);
    auto realizer = centered.new_realize_workspace();
    Eigen::MatrixXd realized;
    realizer->realize_copy(realized);

    auto C = create_random_vector(20, 1234);
    Eigen::VectorXd expected = realized.adjoint() * C;

    auto wrk = centered.new_adjoint_workspace();
    Eigen::VectorXd output(10);
    wrk->multiply(C, output);
    expect_equal_matrix(expected, output);

    // Check that non-zeroed outputs have no effect.
    wrk->multiply(C, output);
    expect_equal_matrix(expected, output);
}

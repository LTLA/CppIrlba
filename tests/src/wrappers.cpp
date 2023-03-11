#include <gtest/gtest.h>
#include "irlba/wrappers.hpp"
#include "Eigen/Dense"
#include "NormalSampler.h"
#include "compare.h"

TEST(WrapperTest, Centering) {
    auto A = create_random_matrix(20, 10);
    auto B = create_random_vector(10);

    auto centered = irlba::Centered(&A, &B);
    EXPECT_EQ(centered.rows(), A.rows());
    EXPECT_EQ(centered.cols(), A.cols());
    auto ref = centered.realize();

    {
        auto C = create_random_vector(10, 1234);
        Eigen::VectorXd output(20);
        centered.multiply(C, output);
        expect_equal_matrix<Eigen::MatrixXd>(ref * C, output);
    }

    {
        auto C = create_random_vector(20, 1234);
        Eigen::VectorXd output(10);
        centered.adjoint_multiply(C, output);
        expect_equal_matrix<Eigen::MatrixXd>(ref.adjoint() * C, output);
    }
}

TEST(WrapperTest, Scaling) {
    auto A = create_random_matrix(20, 10);
    auto B = create_random_vector(10);

    auto centered = irlba::Scaled(&A, &B);
    EXPECT_EQ(centered.rows(), A.rows());
    EXPECT_EQ(centered.cols(), A.cols());
    auto ref = centered.realize();

    {
        auto C = create_random_vector(10, 1234);
        Eigen::VectorXd output(20);
        centered.multiply(C, output);
        expect_equal_matrix<Eigen::MatrixXd>(ref * C, output);
    }

    {
        auto C = create_random_vector(20, 1234);
        Eigen::VectorXd output(10);
        centered.adjoint_multiply(C, output);
        expect_equal_matrix<Eigen::MatrixXd>(ref.adjoint() * C, output);
    }
}

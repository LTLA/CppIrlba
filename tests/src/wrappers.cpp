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
        Eigen::VectorXd expected = ref * C;

        auto wrk = centered.workspace();
        bool is_placeholder = std::is_same<decltype(wrk), bool>::value;
        EXPECT_TRUE(is_placeholder); // just inherits the child.

        Eigen::VectorXd output(20);
        centered.multiply(C, wrk, output);
        expect_equal_matrix(expected, output);

        // Checking that the wrapper method works.
        output.setZero();
        irlba::wrapped_multiply(&centered, C, wrk, output);
        expect_equal_matrix(expected, output);
    }

    {
        auto C = create_random_vector(20, 1234);
        Eigen::VectorXd expected = ref.adjoint() * C;

        auto wrk = centered.adjoint_workspace();
        bool is_placeholder = std::is_same<decltype(wrk), bool>::value;
        EXPECT_TRUE(is_placeholder); // just inherits the child.

        Eigen::VectorXd output(10);
        centered.adjoint_multiply(C, wrk, output);
        expect_equal_matrix(expected, output);

        // Checking that the wrapper method works.
        output.setZero();
        irlba::wrapped_adjoint_multiply(&centered, C, wrk, output);
        expect_equal_matrix(expected, output);
    }
}

TEST(WrapperTest, Scaling) {
    auto A = create_random_matrix(20, 10);
    auto B = create_random_vector(10);

    auto scaled = irlba::Scaled(&A, &B);
    EXPECT_EQ(scaled.rows(), A.rows());
    EXPECT_EQ(scaled.cols(), A.cols());
    auto ref = scaled.realize();

    {
        auto C = create_random_vector(10, 1234);
        Eigen::VectorXd expected = ref * C;

        auto wrk = scaled.workspace();
        bool is_placeholder = std::is_same<decltype(wrk), bool>::value;
        EXPECT_FALSE(is_placeholder); // defines its own workspace. 

        Eigen::VectorXd output(20);
        scaled.multiply(C, wrk, output);
        expect_equal_matrix(expected, output);

        // Checking that the wrapper method works.
        output.setZero();
        irlba::wrapped_multiply(&scaled, C, wrk, output);
        expect_equal_matrix(expected, output);
    }

    {
        auto C = create_random_vector(20, 1234);
        Eigen::VectorXd expected = ref.adjoint() * C;

        auto wrk = scaled.adjoint_workspace();
        bool is_placeholder = std::is_same<decltype(wrk), bool>::value;
        EXPECT_TRUE(is_placeholder); // just inherits the child.

        Eigen::VectorXd output(10);
        scaled.adjoint_multiply(C, wrk, output);
        expect_equal_matrix(expected, output);

        // Checking that the wrapper method works.
        output.setZero();
        irlba::wrapped_adjoint_multiply(&scaled, C, wrk, output);
        expect_equal_matrix(expected, output);
    }
}

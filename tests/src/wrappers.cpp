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

    EXPECT_FALSE(irlba::has_multiply_method<decltype(centered)>::value);
    EXPECT_FALSE(irlba::has_adjoint_multiply_method<decltype(centered)>::value);
    EXPECT_TRUE(irlba::has_realize_method<decltype(centered)>::value);

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

TEST(WrapperTest, ScalingColumnDivide) {
    auto A = create_random_matrix(20, 10);
    auto B = create_random_vector(10);

    auto scaled = irlba::Scaled(&A, &B);
    EXPECT_EQ(scaled.rows(), A.rows());
    EXPECT_EQ(scaled.cols(), A.cols());
    auto ref = scaled.realize();

    EXPECT_FALSE(irlba::has_multiply_method<decltype(scaled)>::value);
    EXPECT_FALSE(irlba::has_adjoint_multiply_method<decltype(scaled)>::value);
    EXPECT_TRUE(irlba::has_realize_method<decltype(scaled)>::value);

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

TEST(WrapperTest, ScalingColumnMultiply) {
    auto A = create_random_matrix(20, 10);
    auto B = create_random_vector(10);

    auto scaled = irlba::Scaled<decltype(A), true, false>(&A, &B);
    EXPECT_EQ(scaled.rows(), A.rows());
    EXPECT_EQ(scaled.cols(), A.cols());
    auto ref = scaled.realize();

    {
        Eigen::VectorXd B0 = (1 / B.array()).matrix();
        auto scaled0 = irlba::Scaled<decltype(A), true, true>(&A, &B0);
        auto ref0 = scaled0.realize();
        expect_equal_matrix(ref, ref0);
    }

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

TEST(WrapperTest, ScalingRowDivide) {
    auto A = create_random_matrix(20, 10);
    auto B = create_random_vector(20);

    auto scaled = irlba::Scaled<decltype(A), false, true>(&A, &B);
    EXPECT_EQ(scaled.rows(), A.rows());
    EXPECT_EQ(scaled.cols(), A.cols());
    auto ref = scaled.realize();

    {
        auto C = create_random_vector(10, 1234);
        Eigen::VectorXd expected = ref * C;

        auto wrk = scaled.workspace();
        bool is_placeholder = std::is_same<decltype(wrk), bool>::value;
        EXPECT_TRUE(is_placeholder); // just inherits the child.

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
        EXPECT_FALSE(is_placeholder); // defines its own workspace. 

        Eigen::VectorXd output(10);
        scaled.adjoint_multiply(C, wrk, output);
        expect_equal_matrix(expected, output);

        // Checking that the wrapper method works.
        output.setZero();
        irlba::wrapped_adjoint_multiply(&scaled, C, wrk, output);
        expect_equal_matrix(expected, output);
    }
}

TEST(WrapperTest, ScalingRowMultiply) {
    auto A = create_random_matrix(20, 10);
    auto B = create_random_vector(20);

    auto scaled = irlba::Scaled<decltype(A), false, false>(&A, &B);
    EXPECT_EQ(scaled.rows(), A.rows());
    EXPECT_EQ(scaled.cols(), A.cols());
    auto ref = scaled.realize();

    {
        Eigen::VectorXd B0 = (1 / B.array()).matrix();
        auto scaled0 = irlba::Scaled<decltype(A), false, true>(&A, &B0);
        auto ref0 = scaled0.realize();
        expect_equal_matrix(ref, ref0);
    }

    {
        auto C = create_random_vector(10, 1234);
        Eigen::VectorXd expected = ref * C;

        auto wrk = scaled.workspace();
        bool is_placeholder = std::is_same<decltype(wrk), bool>::value;
        EXPECT_TRUE(is_placeholder); // just inherits the child

        Eigen::VectorXd output(10);
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
        EXPECT_FALSE(is_placeholder); // defines its own workspace. 

        Eigen::VectorXd output(20);
        scaled.adjoint_multiply(C, wrk, output);
        expect_equal_matrix(expected, output);

        // Checking that the wrapper method works.
        output.setZero();
        irlba::wrapped_adjoint_multiply(&scaled, C, wrk, output);
        expect_equal_matrix(expected, output);
    }
}


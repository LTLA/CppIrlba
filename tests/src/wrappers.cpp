#include <gtest/gtest.h>
#include "irlba/wrappers.hpp"
#include "Eigen/Dense"
#include "NormalSampler.h"
#include "compare.h"

class WrapperCenteringTest : public ::testing::Test {
protected:
    inline static Eigen::MatrixXd A;
    inline static Eigen::VectorXd B;

    static void SetUpTestSuite() {
        A = create_random_matrix(20, 10);
        B = create_random_vector(10);
    }
};

TEST_F(WrapperCenteringTest, Basic) {
    auto centered = irlba::Centered(A, B);
    EXPECT_EQ(centered.rows(), A.rows());
    EXPECT_EQ(centered.cols(), A.cols());

    auto realized = centered.template realize<Eigen::MatrixXd>();
    Eigen::MatrixXd expected = A;
    for (Eigen::Index r = 0; r < A.rows(); ++r) {
        expected.row(r).array() -= B.array();
    }
    expect_equal_matrix(expected, realized);
}

TEST_F(WrapperCenteringTest, Multiply) {
    auto centered = irlba::Centered(A, B);
    auto realized = centered.template realize<Eigen::MatrixXd>();

    auto C = create_random_vector(10, 1234);
    Eigen::VectorXd expected = realized * C;

    auto wrk = centered.workspace();
    bool is_placeholder = std::is_same<decltype(wrk), bool>::value;
    EXPECT_TRUE(is_placeholder); // just inherits the child.

    Eigen::VectorXd output(20);
    centered.multiply(C, wrk, output);
    expect_equal_matrix(expected, output);

    // Checking that the wrapper method works.
    output.setZero();
    irlba::wrapped_multiply(centered, C, wrk, output);
    expect_equal_matrix(expected, output);
}

TEST_F(WrapperCenteringTest, AdjointMultiply) {
    auto centered = irlba::Centered(A, B);
    auto realized = centered.template realize<Eigen::MatrixXd>();

    auto C = create_random_vector(20, 1234);
    Eigen::VectorXd expected = realized.adjoint() * C;

    auto wrk = centered.adjoint_workspace();
    bool is_placeholder = std::is_same<decltype(wrk), bool>::value;
    EXPECT_TRUE(is_placeholder); // just inherits the child.

    Eigen::VectorXd output(10);
    centered.adjoint_multiply(C, wrk, output);
    expect_equal_matrix(expected, output);

    // Checking that the wrapper method works.
    output.setZero();
    irlba::wrapped_adjoint_multiply(centered, C, wrk, output);
    expect_equal_matrix(expected, output);
}

class WrapperScalingTest : public ::testing::TestWithParam<bool> {
protected:
    inline static Eigen::MatrixXd A;
    inline static Eigen::VectorXd B1;
    inline static Eigen::VectorXd B2;

public:
    static void SetUpTestSuite() {
        A = create_random_matrix(20, 10);
        B1 = create_random_vector(10);
        B2 = create_random_vector(20);
    }
};

TEST_P(WrapperScalingTest, Basic) {
    auto divide = GetParam();

    {
        auto scaled = irlba::make_Scaled<true>(A, B1, divide);
        EXPECT_EQ(scaled.rows(), A.rows());
        EXPECT_EQ(scaled.cols(), A.cols());

        auto realized = scaled.template realize<Eigen::MatrixXd>();
        Eigen::MatrixXd expected = A;
        for (Eigen::Index r = 0; r < expected.rows(); ++r) {
            if (divide) {
                expected.row(r).array() /= B1.array();
            } else {
                expected.row(r).array() *= B1.array();
            }
        }
        expect_equal_matrix(expected, realized);
    }

    {
        auto scaled = irlba::make_Scaled<false>(A, B2, divide);
        EXPECT_EQ(scaled.rows(), A.rows());
        EXPECT_EQ(scaled.cols(), A.cols());

        auto realized = scaled.template realize<Eigen::MatrixXd>();
        Eigen::MatrixXd expected = A;
        for (Eigen::Index c = 0; c < expected.cols(); ++c) {
            if (divide) {
                expected.col(c).array() /= B2.array();
            } else {
                expected.col(c).array() *= B2.array();
            }
        }
        expect_equal_matrix(expected, realized);
    }
}

TEST_P(WrapperScalingTest, Multiply) {
    auto divide = GetParam();

    {
        auto scaled = irlba::make_Scaled<true>(A, B1, divide);
        auto realized = scaled.template realize<Eigen::MatrixXd>();
        auto C = create_random_vector(10, 5678);
        Eigen::VectorXd expected = realized * C;

        auto wrk = scaled.workspace();
        bool is_placeholder = std::is_same<decltype(wrk), bool>::value;
        EXPECT_FALSE(is_placeholder); // defines its own workspace. 

        Eigen::VectorXd output(20);
        scaled.multiply(C, wrk, output);
        expect_equal_matrix(expected, output);

        // Checking that the wrapper method works.
        output.setZero();
        irlba::wrapped_multiply(scaled, C, wrk, output);
        expect_equal_matrix(expected, output);
    }

    {
        auto scaled = irlba::make_Scaled<false>(A, B2, divide);
        auto realized = scaled.template realize<Eigen::MatrixXd>();
        auto C = create_random_vector(10, 4567);
        Eigen::VectorXd expected = realized * C;

        auto wrk = scaled.workspace();
        bool is_placeholder = std::is_same<decltype(wrk), bool>::value;
        EXPECT_TRUE(is_placeholder); // inherits the child.

        Eigen::VectorXd output(20);
        scaled.multiply(C, wrk, output);
        expect_equal_matrix(expected, output);

        // Checking that the wrapper method works.
        output.setZero();
        irlba::wrapped_multiply(scaled, C, wrk, output);
        expect_equal_matrix(expected, output);
    }
}

TEST_P(WrapperScalingTest, AdjointMultiply) {
    auto divide = GetParam();

    {
        auto scaled = irlba::make_Scaled<true>(A, B1, divide);
        auto realized = scaled.template realize<Eigen::MatrixXd>();
        auto C = create_random_vector(20, 5678);
        Eigen::VectorXd expected = realized.adjoint() * C;

        auto wrk = scaled.adjoint_workspace();
        bool is_placeholder = std::is_same<decltype(wrk), bool>::value;
        EXPECT_TRUE(is_placeholder); // just inherits the child.

        Eigen::VectorXd output(10);
        scaled.adjoint_multiply(C, wrk, output);
        expect_equal_matrix(expected, output);

        // Checking that the wrapper method works.
        output.setZero();
        irlba::wrapped_adjoint_multiply(scaled, C, wrk, output);
        expect_equal_matrix(expected, output);
    }

    {
        auto scaled = irlba::make_Scaled<false>(A, B2, divide);
        auto realized = scaled.template realize<Eigen::MatrixXd>();
        auto C = create_random_vector(20, 1357);
        Eigen::VectorXd expected = realized.adjoint() * C;

        auto wrk = scaled.adjoint_workspace();
        bool is_placeholder = std::is_same<decltype(wrk), bool>::value;
        EXPECT_FALSE(is_placeholder); // defines its own workspace.

        Eigen::VectorXd output(10);
        scaled.adjoint_multiply(C, wrk, output);
        expect_equal_matrix(expected, output);

        // Checking that the wrapper method works.
        output.setZero();
        irlba::wrapped_adjoint_multiply(scaled, C, wrk, output);
        expect_equal_matrix(expected, output);
    }
}

INSTANTIATE_TEST_SUITE_P(
    Wrappers,
    WrapperScalingTest,
    ::testing::Values(true, false)
);

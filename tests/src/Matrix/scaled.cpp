#include <gtest/gtest.h>

#include "irlba/Matrix/scaled.hpp"
#include "irlba/Matrix/simple.hpp"

#include "../NormalSampler.h"
#include "../compare.h"

#include "Eigen/Dense"

class ScaledTest : public ::testing::TestWithParam<bool> {
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

TEST_P(ScaledTest, Basic) {
    auto divide = GetParam();
    irlba::SimpleMatrix<Eigen::VectorXd, Eigen::MatrixXd, decltype(&A)> wrapped(&A);

    {
        irlba::ScaledMatrix<Eigen::VectorXd, Eigen::MatrixXd> scaled(&wrapped, &B1, true, divide);
        EXPECT_EQ(scaled.rows(), A.rows());
        EXPECT_EQ(scaled.cols(), A.cols());

        auto realizer = scaled.new_realize_workspace();
        Eigen::MatrixXd realized;
        realizer->realize_copy(realized);

        Eigen::MatrixXd expected = A;
        for (Eigen::Index c = 0; c < A.cols(); ++c) {
            for (Eigen::Index r = 0; r < A.rows(); ++r) {
                if (divide) {
                    expected(r, c) /= B1[c];
                } else {
                    expected(r, c) *= B1[c];
                }
            }
        }
        expect_equal_matrix(expected, realized);
    }

    {
        irlba::ScaledMatrix<Eigen::VectorXd, Eigen::MatrixXd> scaled(&wrapped, &B2, false, divide);
        EXPECT_EQ(scaled.rows(), A.rows());
        EXPECT_EQ(scaled.cols(), A.cols());

        auto realizer = scaled.new_realize_workspace();
        Eigen::MatrixXd realized;
        realizer->realize_copy(realized);

        Eigen::MatrixXd expected = A;
        for (Eigen::Index c = 0; c < A.cols(); ++c) {
            for (Eigen::Index r = 0; r < A.rows(); ++r) {
                if (divide) {
                    expected(r, c) /= B2[r];
                } else {
                    expected(r, c) *= B2[r];
                }
            }
        }
        expect_equal_matrix(expected, realized);
    }
}

TEST_P(ScaledTest, Multiply) {
    auto divide = GetParam();
    irlba::SimpleMatrix<Eigen::VectorXd, Eigen::MatrixXd, decltype(&A)> wrapped(&A);

    {
        auto scaled = irlba::ScaledMatrix<Eigen::VectorXd, Eigen::MatrixXd>(&wrapped, &B1, true, divide);
        auto realizer = scaled.new_realize_workspace();
        Eigen::MatrixXd realized;
        realizer->realize_copy(realized);

        auto C = create_random_vector(10, 5678);
        Eigen::VectorXd expected = realized * C;

        auto wrk = scaled.new_workspace();
        Eigen::VectorXd output(20);
        wrk->multiply(C, output);
        expect_equal_matrix(expected, output);
    }

    {
        irlba::ScaledMatrix<Eigen::VectorXd, Eigen::MatrixXd> scaled(&wrapped, &B2, false, divide);
        auto realizer = scaled.new_realize_workspace();
        Eigen::MatrixXd realized;
        realizer->realize_copy(realized);

        auto C = create_random_vector(10, 4567);
        Eigen::VectorXd expected = realized * C;

        auto wrk = scaled.new_workspace();
        Eigen::VectorXd output(20);
        wrk->multiply(C, output);
        expect_equal_matrix(expected, output);
    }
}

TEST_P(ScaledTest, AdjointMultiply) {
    auto divide = GetParam();
    irlba::SimpleMatrix<Eigen::VectorXd, Eigen::MatrixXd, decltype(&A)> wrapped(&A);

    {
        irlba::ScaledMatrix<Eigen::VectorXd, Eigen::MatrixXd> scaled(&wrapped, &B1, true, divide);
        auto realizer = scaled.new_realize_workspace();
        Eigen::MatrixXd realized;
        realizer->realize_copy(realized);

        auto C = create_random_vector(20, 5678);
        Eigen::VectorXd expected = realized.adjoint() * C;

        auto wrk = scaled.new_adjoint_workspace();
        Eigen::VectorXd output(10);
        wrk->multiply(C, output);
        expect_equal_matrix(expected, output);
    }

    {
        irlba::ScaledMatrix<Eigen::VectorXd, Eigen::MatrixXd> scaled(&wrapped, &B2, false, divide);
        auto realizer = scaled.new_realize_workspace();
        Eigen::MatrixXd realized;
        realizer->realize_copy(realized);

        auto C = create_random_vector(20, 1357);
        Eigen::VectorXd expected = realized.adjoint() * C;

        auto wrk = scaled.new_adjoint_workspace();
        Eigen::VectorXd output(10);
        wrk->multiply(C, output);
        expect_equal_matrix(expected, output);
    }
}

INSTANTIATE_TEST_SUITE_P(
    Scaled,
    ScaledTest,
    ::testing::Values(true, false)
);

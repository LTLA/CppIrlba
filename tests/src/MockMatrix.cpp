#include <gtest/gtest.h>

#include "irlba/wrappers.hpp"
#include "irlba/MockMatrix.hpp"
#include "Eigen/Dense"

#include "NormalSampler.h"
#include "compare.h"

TEST(MockMatrix, Basic) {
    auto A = create_random_matrix(20, 10, 9999);

    irlba::MockMatrix tmp(A);
    EXPECT_EQ(tmp.rows(), A.rows());
    EXPECT_EQ(tmp.cols(), A.cols());

    {
        auto C = create_random_vector(10, 11111);
        auto wrk = wrapped_workspace(tmp);
        Eigen::VectorXd D;
        irlba::wrapped_multiply(tmp, C, wrk, D);

        Eigen::VectorXd ref = A * C;
        expect_equal_matrix(D, ref);
    }

    {
        auto C = create_random_vector(20, 11111);
        auto awrk = wrapped_adjoint_workspace(tmp);
        Eigen::VectorXd D;
        irlba::wrapped_adjoint_multiply(tmp, C, awrk, D);

        Eigen::VectorXd aref = A.adjoint() * C;
        expect_equal_matrix(D, aref);
    }
}


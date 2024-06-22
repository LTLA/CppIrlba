#include <gtest/gtest.h>
#include "irlba/utils.hpp"
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "NormalSampler.h"

TEST(UtilsTest, NormalSampler) {
    NormalSampler norm(10);

    // Different results.
    double first = norm();
    double second = norm();
    EXPECT_NE(first, second);

    // Same results.
    NormalSampler norm2(10);
    double first2 = norm2();
    EXPECT_EQ(first, first2);
}

TEST(UtilsTest, FillNormals) {
    Eigen::VectorXd test(10);
    test.setZero();
    std::mt19937_64 eng(1000);

    // Filled with non-zeros.
    irlba::internal::fill_with_random_normals(test, eng);
    for (auto v : test) {
        EXPECT_NE(v, 0);
    }
    
    std::sort(test.begin(), test.end());
    for (size_t v = 1; v < test.size(); ++v) {
        EXPECT_NE(test[v], test[v-1]);
    }

    // Works for odd sizes.
    Eigen::VectorXd test2(11);
    test2.setZero();

    irlba::internal::fill_with_random_normals(test2, eng);
    for (auto v : test2) {
        EXPECT_NE(v, 0);
    }

    std::sort(test2.begin(), test2.end());
    for (size_t v = 1; v < test2.size(); ++v) {
        EXPECT_NE(test2[v], test2[v-1]);
    }

    // Works for matrices.
    Eigen::MatrixXd test3(13, 2);
    test3.setZero();

    irlba::internal::fill_with_random_normals(test3, 0, eng);
    auto first = test3.col(0);
    for (auto v : first) {
        EXPECT_NE(v, 0);
    }

    auto second = test3.col(1);
    for (auto v : second) {
        EXPECT_EQ(v, 0);
    }
}

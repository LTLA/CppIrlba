#ifndef COMPARE_H
#define COMPARE_H

#include <gtest/gtest.h>
#include "Eigen/Dense"

inline bool same_same(double left, double right, double tol) {
    return std::abs(left - right) <= (std::abs(left) + std::abs(right)) * tol;
}

template<bool centered=false>
void expect_equal_column_vectors(const Eigen::MatrixXd& left, const Eigen::MatrixXd& right, double tol=1e-8) {
    ASSERT_EQ(left.cols(), right.cols());
    ASSERT_EQ(left.rows(), right.rows());

    for (size_t i = 0; i < left.cols(); ++i) {
        for (size_t j = 0; j < left.rows(); ++j) {
            EXPECT_TRUE(same_same(std::abs(left(j, i)), std::abs(right(j, i)), tol));
        }

        double left_sum = std::abs(left.col(i).sum());
        double right_sum = std::abs(right.col(i).sum());
        if constexpr(centered) {
            EXPECT_TRUE(left_sum < tol);
            EXPECT_TRUE(right_sum < tol);
        } else {
            EXPECT_TRUE(same_same(left_sum, std::abs(right_sum), tol));
        }
    }
    return;
}

inline void expect_equal_vectors(const Eigen::VectorXd& left, const Eigen::VectorXd& right, double tol=1e-8) {
    ASSERT_EQ(left.size(), right.size());
    for (size_t i = 0; i < left.size(); ++i) {
        EXPECT_TRUE(same_same(left[i], right[i], tol));
    }
    return;
}

#endif

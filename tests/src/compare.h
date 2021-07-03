#ifndef COMPARE_H
#define COMPARE_H

#include <gtest/gtest.h>
#include "Eigen/Dense"

inline void expect_equal_column_vectors(const Eigen::MatrixXd& left, const Eigen::MatrixXd& right) {
    ASSERT_EQ(left.cols(), right.cols());
    ASSERT_EQ(left.rows(), right.rows());

    for (size_t i = 0; i < left.cols(); ++i) {
        for (size_t j = 0; j < left.rows(); ++j) {
            EXPECT_FLOAT_EQ(std::abs(left(j, i)), std::abs(right(j, i)));
        }
        EXPECT_FLOAT_EQ(std::abs(left.col(i).sum()), std::abs(right.col(i).sum()));
    }
    return;
}

inline void expect_equal_vectors(const Eigen::VectorXd& left, const Eigen::VectorXd& right) {
    ASSERT_EQ(left.size(), right.size());
    for (size_t i = 0; i < left.size(); ++i) {
        EXPECT_FLOAT_EQ(left[i], right[i]);
    }
    return;
}

#endif

#ifndef IRLBA_WRAPPERS_HPP
#define IRLBA_WRAPPERS_HPP

#include "utils.hpp"
#include "Eigen/Dense"

/**
 * @file wrappers.hpp
 *
 * @brief Wrapper classes for multiplication of modified matrices.
 *
 * The idea is to compute the product of a modified matrix with a vector - but without actually modifying the underlying matrix.
 * This is especially important when the modification results in an unnecessary copy and/or loss of sparsity.
 * We achieve this effect by deferring the modification into the subspace defined by vector.
 *
 * An instance `mat` of a wrapper class should implement:
 *
 * - `mat.rows()`, returning the number of rows.
 * - `mat.cols()`, returning the number of columns.
 * - `mat.multiply(rhs, out)`, which computes the matrix product `mat * rhs` and stores it in `out`.
 * `rhs` should be an `Eigen::VectorXd` (or an expression equivalent) while `out` should be a `Eigen::VectorXd`.
 * - `mat.adjoint_multiply(rhs, out)`, which computes the matrix product `mat.adjoint() * rhs` and stores it in `out`.
 * `rhs` should be an `Eigen::VectorXd` (or an expression equivalent) while `out` should be a `Eigen::VectorXd`.
 * - `mat.realize()`, which returns an `Eigen::MatrixXd` containing the matrix with all modifications applied.
 */

namespace irlba {

/**
 * @brief Wrapper for a centered matrix.
 *
 * @tparam Matrix An **Eigen** matrix class - or alternatively, a wrapper class around such a class.
 * 
 * This modification involves centering all columns, i.e., subtracting the mean of each column from the values of that column.
 * Naively doing such an operation would involve loss of sparsity, which we avoid by deferring the subtraction into the subspace defined by `rhs`.
 */
template<class Matrix>
struct Centered {
    /**
     * @param m Underlying matrix to be column-centered.
     * @param c Vector of length equal to the number of columns of `m`,
     * containing the value to subtract from each column.
     */
    Centered(const Matrix* m, const Eigen::VectorXd* c) : mat(m), center(c) {}

    /**
     * @return Number of rows in the matrix.
     */
    auto rows() const { return mat->rows(); }

    /**
     * @return Number of columns in the matrix.
     */
    auto cols() const { return mat->cols(); }

    /**
     * @tparam Right An `Eigen::VectorXd` or equivalent expression.
     *
     * @param[in] rhs The right-hand side of the matrix product.
     * This should be a vector or have only one column.
     * @param[out] out The output vector to store the matrix product.
     * 
     * @return `out` is filled with the product of this matrix and `rhs`.
     */
    template<class Right>
    void multiply(const Right& rhs, Eigen::VectorXd& out) const {
        if constexpr(has_multiply_method<Matrix>::value) {
            out.noalias() = *mat * rhs;
        } else {
            mat->multiply(rhs, out);
        }

        double beta = rhs.dot(*center);
        for (auto& o : out) {
            o -= beta;
        }
        return;
    }

    /**
     * @tparam Right An `Eigen::VectorXd` or equivalent expression.
     *
     * @param[in] rhs The right-hand side of the matrix product.
     * This should be a vector or have only one column.
     * @param[out] out The output vector to store the matrix product.
     * 
     * @return `out` is filled with the product of the transpose of this matrix and `rhs`.
     */
    template<class Right>
    void adjoint_multiply(const Right& rhs, Eigen::VectorXd& out) const {
        if constexpr(has_adjoint_multiply_method<Matrix>::value) {
            out.noalias() = mat->adjoint() * rhs;
        } else {
            mat->adjoint_multiply(rhs, out);
        }

        double beta = rhs.sum();
        out -= beta * (*center);
        return;
    }

    /**
     * @return A realized version of the centered matrix,
     * where the centering has been explicitly applied.
     */
    Eigen::MatrixXd realize() const {
        auto subtractor = [&](Eigen::MatrixXd& m) -> void {
            for (Eigen::Index c = 0; c < m.cols(); ++c) {
                for (Eigen::Index r = 0; r < m.rows(); ++r) {
                    m(r, c) -= (*center)[c];
                }
            }
        };

        if constexpr(has_realize_method<Matrix>::value) {
            Eigen::MatrixXd output = mat->realize();
            subtractor(output);
            return output;
        } else {
            Eigen::MatrixXd output(*mat);
            subtractor(output);
            return output;
        }
    }

private:
    const Matrix* mat;
    const Eigen::VectorXd* center;
};

/**
 * @brief Wrapper for a scaled matrix.
 *
 * @tparam Matrix An **Eigen** matrix class - or alternatively, a wrapper class around such a class.
 * 
 * This modification involves scaling all columns, i.e., dividing the values of each column by the standard deviation of that column to achieve unit variance.
 * Naively doing such an operation would involve a copy of the matrix, which we avoid by deferring the scaling into the subspace defined by `rhs`.
 */
template<class Matrix>
struct Scaled {
    /**
     * @param m Underlying matrix to be column-scaled.
     * @param s Vector of length equal to the number of columns of `m`,
     * containing the value to scale each column.
     */
    Scaled(const Matrix* m, const Eigen::VectorXd* s) : mat(m), scale(s) {}

    /**
     * @return Number of rows in the matrix.
     */
    auto rows() const { return mat->rows(); }

    /**
     * @return Number of columns in the matrix.
     */
    auto cols() const { return mat->cols(); }

    /**
     * @tparam Right An `Eigen::VectorXd` or equivalent expression.
     *
     * @param[in] rhs The right-hand side of the matrix product.
     * This should be a vector or have only one column.
     * @param[out] out The output vector to store the matrix product.
     * 
     * @return `out` is filled with the product of this matrix and `rhs`.
     */
    template<class Right>
    void multiply(const Right& rhs, Eigen::VectorXd& out) const {
        if constexpr(has_multiply_method<Matrix>::value) {
            out.noalias() = *mat * rhs.cwiseQuotient(*scale);
        } else {
            mat->multiply(rhs.cwiseQuotient(*scale), out);
        }
        return;
    }

    /**
     * @tparam Right An `Eigen::VectorXd` or equivalent expression.
     *
     * @param[in] rhs The right-hand side of the matrix product.
     * This should be a vector or have only one column.
     * @param[out] out The output vector to store the matrix product.
     * 
     * @return `out` is filled with the product of the transpose of this matrix and `rhs`.
     */
    template<class Right>
    void adjoint_multiply(const Right& rhs, Eigen::VectorXd& out) const {
        if constexpr(has_adjoint_multiply_method<Matrix>::value) {
            out.noalias() = mat->adjoint() * rhs;
        } else {
            mat->adjoint_multiply(rhs, out);
        }
        out.noalias() = out.cwiseQuotient(*scale);
        return;
    }

    /**
     * @return A realized version of the scaled matrix,
     * where the scaling has been explicitly applied.
     */
    Eigen::MatrixXd realize() const {
        auto scaler = [&](Eigen::MatrixXd& m) -> void {
            for (Eigen::Index c = 0; c < m.cols(); ++c) {
                for (Eigen::Index r = 0; r < m.rows(); ++r) {
                    m(r, c) /= (*scale)[c];
                }
            }
        };

        if constexpr(has_realize_method<Matrix>::value) {
            Eigen::MatrixXd output = mat->realize();
            scaler(output);
            return output;
        } else {
            Eigen::MatrixXd output(*mat);
            scaler(output);
            return output;
        }
    }

private:
    const Matrix* mat;
    const Eigen::VectorXd* scale;
};

}

#endif

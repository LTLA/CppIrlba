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
 * - `mat.workspace()`, returning an instance of a workspace class for multiplication.
 * - `mat.adjoint_workspace()`, returning an instance of a workspace class for adjoint multiplication.
 * - `mat.multiply(rhs, work, out)`, which computes the matrix product `mat * rhs` and stores it in `out` - see `irlba::Centered::multiply()` for the typical signature.
 * `rhs` should be a const reference to an `Eigen::VectorXd` (or an expression equivalent, via templating) while `out` should be a non-const reference to a `Eigen::VectorXd`.
 * `work` should be the return value of `mat.workspace()` and is passed in as a non-const reference.
 * - `mat.adjoint_multiply(rhs, work, out)`, which computes the matrix product `mat.adjoint() * rhs` and stores it in `out` - see `irlba::Centered::adjoint_multiply()` for the typical signature.
 * `rhs` should be a const reference to an `Eigen::VectorXd` (or an expression equivalent, via templating) while `out` should be a non-const reference to a `Eigen::VectorXd`.
 * `work` should be the return value of `mat.adjoint_workspace()` and is passed in as a non-const reference.
 * - `mat.realize()`, which returns an `Eigen::MatrixXd` containing the matrix with all modifications applied.
 *
 * The workspace class is used to allocate space for intermediate results across multiple calls to `multiply()`.
 * This class should contain a member of type `WrappedWorkspace<M>`, where `M` is the type of the underlying matrix;
 * this member can be initialized by calling the `wrapped_workspace()` function on the underlying matrix.
 * If a wrapper does not have any intermediate results, it can just return `WrappedWorkspace<M>` directly.
 * The same logic applies to `adjoint_multiply()` using the `AdjointWrappedWorkspace` template class and `wrapped_adjoint_workspace()`.
 *
 * Implementations of the `multiply()` and `adjoint_multiply()` methods may use the `wrapped_multiply()` and `wrapped_adjoint_multiply()` functions.
 * This will handle the differences in the calls between **Eigen** matrices and **irlba** wrappers.
 */

namespace irlba {

/**
 * @cond
 */
template<class Matrix, typename = int>
struct WrappedWorkspaceInternal {
    typedef bool type;
};

template<class Matrix>
struct WrappedWorkspaceInternal<Matrix, decltype((void) std::declval<Matrix>().workspace(), 0)> {
    typedef decltype(std::declval<Matrix>().workspace()) type;
};

template<class Matrix, typename = int>
struct WrappedAdjointWorkspaceInternal {
    typedef bool type;
};

template<class Matrix>
struct WrappedAdjointWorkspaceInternal<Matrix, decltype((void) std::declval<Matrix>().adjoint_workspace(), 0)> {
    typedef decltype(std::declval<Matrix>().adjoint_workspace()) type;
};
/**
 * @endcond
 */

/**
 * @tparam Matrix Type of the underlying matrix in the wrapper.
 *
 * This type is equivalent to the workspace class of `Matrix`, or a placeholder boolean if `Matrix` is an Eigen class.
 */
template<class Matrix>
using WrappedWorkspace = typename WrappedWorkspaceInternal<Matrix>::type;

/**
 * @tparam Matrix Type of the underlying matrix in the wrapper.
 *
 * This type is equivalent to the adjoint workspace class of `Matrix`, or a placeholder boolean if `Matrix` is an Eigen class.
 */
template<class Matrix>
using WrappedAdjointWorkspace = typename WrappedAdjointWorkspaceInternal<Matrix>::type;

/**
 * @tparam Matrix Type of the underlying matrix in the wrapper.
 * @param mat Pointer to the wrapped matrix instance.
 * @return The workspace of `mat`, or `false` if `Matrix` is an **Eigen** class.
 */
template<class Matrix>
WrappedWorkspace<Matrix> wrapped_workspace(const Matrix* mat) {
    if constexpr(has_multiply_method<Matrix>::value) { // using this as a proxy for whether it's an Eigen matrix or not.
        return false;
    } else {
        return mat->workspace();
    }
}

/**
 * @tparam Matrix Type of the underlying matrix in the wrapper.
 * @param mat Pointer to the wrapped matrix instance.
 * @return The adjoint workspace of `mat`, or `false` if `Matrix` is an **Eigen** class.
 */
template<class Matrix>
WrappedAdjointWorkspace<Matrix> wrapped_adjoint_workspace(const Matrix* mat) {
    if constexpr(has_adjoint_multiply_method<Matrix>::value) {
        return false;
    } else {
        return mat->adjoint_workspace();
    }
}

/**
 * @tparam Matrix Type of the wrapped matrix.
 * @tparam Right An `Eigen::VectorXd` or equivalent expression.
 *
 * @param[in] mat Pointer to the wrapped matrix instance.
 * @param[in] rhs The right-hand side of the matrix product.
 * @param work The return value of `wrapped_workspace()` on `mat`.
 * @param[out] out The output vector to store the matrix product.
 * This is filled with the product of this matrix and `rhs`.
 */
template<class Matrix, class Right>
void wrapped_multiply(const Matrix* mat, const Right& rhs, WrappedWorkspace<Matrix>& work, Eigen::VectorXd& out) {
    if constexpr(has_multiply_method<Matrix>::value) {
        out.noalias() = *mat * rhs;
    } else {
        mat->multiply(rhs, work, out);
    }
}

/**
 * @tparam Matrix Type of the wrapped matrix.
 * @tparam Right An `Eigen::VectorXd` or equivalent expression.
 *
 * @param[in] mat Poitner to the wrapped matrix instance.
 * @param[in] rhs The right-hand side of the matrix product.
 * @param work The return value of `wrapped_adjoint_workspace()` on `mat`.
 * @param[out] out The output vector to store the matrix product.
 * This is filled with the product of this matrix and `rhs`.
 */
template<class Matrix, class Right>
void wrapped_adjoint_multiply(const Matrix* mat, const Right& rhs, WrappedAdjointWorkspace<Matrix>& work, Eigen::VectorXd& out) {
    if constexpr(has_adjoint_multiply_method<Matrix>::value) {
        out.noalias() = mat->adjoint() * rhs;
    } else {
        mat->adjoint_multiply(rhs, work, out);
    }
}

/**
 * @tparam Matrix Type of the wrapped matrix.
 * @param[in] mat Pointer to the wrapped matrix instance.
 * @return A dense **Eigen** matrix containing the realized contents of `mat`.
 */
template<class Matrix>
Eigen::MatrixXd wrapped_realize(const Matrix* mat) {
    if constexpr(has_realize_method<Matrix>::value) {
        return mat->realize();
    } else {
        return Eigen::MatrixXd(*mat);
    }
}

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

public:
    /**
     * Workspace type for `multiply()`.
     * Currently, this is just an alias for the workspace type of the underlying matrix.
     */
    typedef WrappedWorkspace<Matrix> Workspace;

    /**
     * @return Workspace for use in `multiply()`.
     */
    Workspace workspace() const {
        return wrapped_workspace(mat);
    }

    /**
     * Workspace type for `adjoint_multiply()`.
     * Currently, this is just an alias for the adjoint workspace type of the underlying matrix.
     */
    typedef WrappedAdjointWorkspace<Matrix> AdjointWorkspace;

    /**
     * @return Workspace for use in `adjoint_multiply()`.
     */
    AdjointWorkspace adjoint_workspace() const {
        return wrapped_adjoint_workspace(mat);
    }

public:
    /**
     * @tparam Right An `Eigen::VectorXd` or equivalent expression.
     *
     * @param[in] rhs The right-hand side of the matrix product.
     * @param work The return value of `workspace()`.
     * This can be reused across multiple `multiply()` calls.
     * @param[out] out The output vector to store the matrix product.
     * This is filled with the product of this matrix and `rhs`.
     */
    template<class Right>
    void multiply(const Right& rhs, Workspace& work, Eigen::VectorXd& out) const {
        wrapped_multiply(mat, rhs, work, out);
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
     * @param work The return value of `adjoint_workspace()`.
     * This can be reused across multiple `adjoint_multiply()` calls.
     * @param[out] out The output vector to store the matrix product.
     * This is filled with the product of the transpose of this matrix and `rhs`.
     */
    template<class Right>
    void adjoint_multiply(const Right& rhs, AdjointWorkspace& work, Eigen::VectorXd& out) const {
        wrapped_adjoint_multiply(mat, rhs, work, out);
        double beta = rhs.sum();
        out -= beta * (*center);
        return;
    }

    /**
     * @return A realized version of the centered matrix,
     * where the centering has been explicitly applied.
     */
    Eigen::MatrixXd realize() const {
        Eigen::MatrixXd output = wrapped_realize(mat);
        for (Eigen::Index c = 0; c < output.cols(); ++c) {
            for (Eigen::Index r = 0; r < output.rows(); ++r) {
                output(r, c) -= (*center)[c];
            }
        }
        return output;
    }

private:
    const Matrix* mat;
    const Eigen::VectorXd* center;
};

/**
 * @brief Wrapper for a scaled matrix.
 *
 * @tparam Matrix_ The underlying **Eigen** matrix class - or alternatively, a wrapper class around such a class.
 * @tparam column_ Whether to scale the columns.
 * If `false`, scaling is applied to the rows instead.
 * @tparam divide_ Whether to divide by the supplied scaling factors.
 * 
 * This modification involves scaling all rows or columns, i.e., multiplying or dividing the values of each row/column by some arbitrary value.
 * For example, we can use this to divide each column by the standard deviation to achieve unit variance in principal components analyses.
 * Naively doing such an operation would involve a copy of the matrix, which we avoid by deferring the scaling into the subspace defined by `rhs`.
 */
template<class Matrix_, bool column_ = true, bool divide_ = true>
struct Scaled {
    /**
     * @param m Underlying matrix to be column-scaled (if `column_ = true`) or row-scaled (otherwise).
     * @param s Vector of length equal to the number of columns (if `column_ = true`) or rows (otherwise) of `m`,
     * containing the scaling factor to divide (if `divide_ = true`) or multiply (otherwise) to each column/row.
     */
    Scaled(const Matrix_* m, const Eigen::VectorXd* s) : mat(m), scale(s) {}

    /**
     * @cond
     */
    auto rows() const { return mat->rows(); }

    auto cols() const { return mat->cols(); }
    /**
     * @endcond
     */

public:
    /**
     * @cond
     */
    template<template<class> class Wrapper_>
    struct BufferedWorkspace {
        BufferedWorkspace(size_t n, Wrapper_<Matrix_> c) : buffer(n), child(std::move(c)) {}
        Eigen::VectorXd buffer;
        Wrapper_<Matrix_> child;
    };

    typedef typename std::conditional<column_, BufferedWorkspace<WrappedWorkspace>, WrappedWorkspace<Matrix_> >::type Workspace;

    Workspace workspace() const {
        if constexpr(column_) {
            return BufferedWorkspace<WrappedWorkspace>(mat->cols(), wrapped_workspace(mat));
        } else {
            return wrapped_workspace(mat);
        }
    }

    typedef typename std::conditional<column_, WrappedAdjointWorkspace<Matrix_>, BufferedWorkspace<WrappedAdjointWorkspace> >::type AdjointWorkspace;

    AdjointWorkspace adjoint_workspace() const {
        if constexpr(column_) {
            return wrapped_adjoint_workspace(mat);
        } else {
            return BufferedWorkspace<WrappedAdjointWorkspace>(mat->rows(), wrapped_adjoint_workspace(mat));
        }
    }
    /**
     * @endcond
     */

public:
    /**
     * @cond
     */
    template<class Right_>
    void multiply(const Right_& rhs, Workspace& work, Eigen::VectorXd& out) const {
        if constexpr(column_) {
            if constexpr(divide_) {
                // We store the result here, because the underlying matrix's multiply()
                // might need to access rhs/scale multiple times, especially if it's
                // parallelized. Better to pay the cost of accessing a separate memory
                // space than computing the quotient repeatedly.
                work.buffer = rhs.cwiseQuotient(*scale);
            } else {
                work.buffer = rhs.cwiseProduct(*scale);
            }
            wrapped_multiply(mat, work.buffer, work.child, out);

        } else {
            wrapped_multiply(mat, rhs, work, out);
            if constexpr(divide_) {
                out.array() /= scale->array();
            } else {
                out.array() *= scale->array();
            }
        }
    }

    template<class Right_>
    void adjoint_multiply(const Right_& rhs, AdjointWorkspace& work, Eigen::VectorXd& out) const {
        if constexpr(column_) {
            wrapped_adjoint_multiply(mat, rhs, work, out);
            if constexpr(divide_) {
                out.array() /= scale->array();
            } else {
                out.array() *= scale->array();
            }

        } else {
            if constexpr(divide_) {
                work.buffer = rhs.cwiseQuotient(*scale);
            } else {
                work.buffer = rhs.cwiseProduct(*scale);
            }
            wrapped_adjoint_multiply(mat, work.buffer, work.child, out);
        }
    }

    Eigen::MatrixXd realize() const {
        Eigen::MatrixXd output = wrapped_realize(mat);

        for (Eigen::Index c = 0; c < output.cols(); ++c) {
            for (Eigen::Index r = 0; r < output.rows(); ++r) {
                if constexpr(column_) {
                    if constexpr(divide_) {
                        output(r, c) /= (*scale)[c];
                    } else {
                        output(r, c) *= (*scale)[c];
                    }
                } else {
                    if constexpr(divide_) {
                        output(r, c) /= (*scale)[r];
                    } else {
                        output(r, c) *= (*scale)[r];
                    }
                }
            }
        }

        return output;
    }
    /**
     * @endcond
     */

private:
    const Matrix_* mat;
    const Eigen::VectorXd* scale;
};

}

#endif

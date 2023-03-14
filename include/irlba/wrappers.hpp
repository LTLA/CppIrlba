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
 * - `mat.multiply(rhs, work, out)`, which computes the matrix product `mat * rhs` and stores it in `out` - see `Centered::multiply()` for the typical signature.
 * `rhs` should be a const reference to an `Eigen::VectorXd` (or an expression equivalent, via templating) while `out` should be a non-const reference to a `Eigen::VectorXd`.
 * `work` should be the return value of `mat.workspace()` and is passed in as a non-const reference.
 * - `mat.adjoint_multiply(rhs, work, out)`, which computes the matrix product `mat.adjoint() * rhs` and stores it in `out` - see `Centered::adjoint_multiply()` for the typical signature.
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
 * 
 * @return `out` is filled with the product of this matrix and `rhs`.
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
 * 
 * @return `out` is filled with the product of this matrix and `rhs`.
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
     * 
     * @return `out` is filled with the product of this matrix and `rhs`.
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
     * 
     * @return `out` is filled with the product of the transpose of this matrix and `rhs`.
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

public:
    /**
     * Workspace type for `multiply()`.
     */
    struct Workspace {
        /**
         * @cond
         */
        Workspace(size_t n, WrappedWorkspace<Matrix> c) : product(n), child(std::move(c)) {}
        Eigen::VectorXd product;
        WrappedWorkspace<Matrix> child;
        /**
         * @endcond
         */
    };

    /**
     * @return Workspace for use in `multiply()`.
     */
    Workspace workspace() const {
        return Workspace(mat->cols(), wrapped_workspace(mat));
    }

    /**
     * Workspace type for `adjoint_multiply()`.
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
     * This should be a vector or have only one column.
     * @param work The return value of `workspace()`.
     * This can be reused across multiple `multiply()` calls.
     * @param[out] out The output vector to store the matrix product.
     * 
     * @return `out` is filled with the product of this matrix and `rhs`.
     */
    template<class Right>
    void multiply(const Right& rhs, Workspace& work, Eigen::VectorXd& out) const {
        work.product = rhs.cwiseQuotient(*scale);
        wrapped_multiply(mat, work.product, work.child, out);
        return;
    }

    /**
     * @tparam Right An `Eigen::VectorXd` or equivalent expression.
     *
     * @param[in] rhs The right-hand side of the matrix product.
     * This should be a vector or have only one column.
     * @param work The return value of `adjoint_workspace()`.
     * This can be reused across multiple `adjoint_multiply()` calls.
     * @param[out] out The output vector to store the matrix product.
     * 
     * @return `out` is filled with the product of the transpose of this matrix and `rhs`.
     */
    template<class Right>
    void adjoint_multiply(const Right& rhs, AdjointWorkspace& work, Eigen::VectorXd& out) const {
        wrapped_adjoint_multiply(mat, rhs, work, out);
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

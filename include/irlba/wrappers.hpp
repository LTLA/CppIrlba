#ifndef IRLBA_WRAPPERS_HPP
#define IRLBA_WRAPPERS_HPP

#include "utils.hpp"
#include "Eigen/Dense"
#include <type_traits>

/**
 * @file wrappers.hpp
 * @brief Wrapper classes for multiplication of modified matrices.
 */

namespace irlba {

// This painful setup for the workspaces is because std::conditional requires
// both options to be valid types, so we can't just make a conditional to
// extract Matrix_::(Adjoint)Workspace for non-Eigen matrices (as this won't be
// valid for Eigen matrices that lack the typedef).

/**
 * @brief Get the type of workspace for `wrapped_multiply()`.
 * @tparam Matrix_ Class satisfying the `MockMatrix` interface.
 */
template<class Matrix_, typename = int>
struct get_workspace {
    /**
     * Alias for the workspace type.
     */
    typedef typename Matrix_::Workspace type;
};

/**
 * @brief Get the type of workspace for `wrapped_multiply()`.
 * @tparam Matrix_ A floating-point `Eigen::Matrix` class.
 * This is detected based on the presence of an `Index` typedef.
 */
template<class Matrix_>
struct get_workspace<Matrix_, decltype((void) std::declval<typename Matrix_::Index>(), 0)> {
    /**
     * Placeholder boolean.
     */
    typedef bool type;
};

/**
 * Type of workspace for `wrapped_multiply()`.
 * This can be used to satisfy the `MockMatrix::Workspace` interface,
 * either directly or as a component of a larger workspace class.
 *
 * @tparam Matrix_ A floating-point `Eigen::Matrix` class, or a class satisfying the `MockMatrix` interface.
 */
template<class Matrix_>
using WrappedWorkspace = typename get_workspace<Matrix_>::type;

/**
 * @tparam Matrix_ A floating-point `Eigen::Matrix` class, or a class satisfying the `MockMatrix` interface.
 * @param matrix Instance of a matrix.
 * @return Workspace to use in `multiply()`.
 */
template<class Matrix_>
WrappedWorkspace<Matrix_> wrapped_workspace(const Matrix_& matrix) {
    if constexpr(std::is_same<WrappedWorkspace<Matrix_>, bool>::value) {
        return false;
    } else {
        return matrix.workspace();
    }
}

/**
 * @brief Get the type of workspace for `wrapped_adjoint_multiply()`.
 * @tparam Matrix_ Class satisfying the `MockMatrix` interface.
 */
template<class Matrix_, typename = int>
struct get_adjoint_workspace {
    /**
     * Alias for the workspace type.
     */
    typedef typename Matrix_::AdjointWorkspace type;
};

/**
 * @brief Get the type of workspace for `wrapped_adjoint_multiply()`.
 * @tparam Matrix_ A floating-point `Eigen::Matrix` class.
 * This is detected based on the presence of an `Index` typedef.
 */
template<class Matrix_>
struct get_adjoint_workspace<Matrix_, decltype((void) std::declval<typename Matrix_::Index>(), 0)> {
    /**
     * Placeholder boolean.
     */
    typedef bool type;
};

/**
 * Type of workspace for `wrapped_adjoint_multiply()`.
 * This can be used to satisfy the `MockMatrix::AdjointWorkspace` interface,
 * either directly or as a component of a larger workspace class.
 *
 * @tparam Matrix_ A floating-point `Eigen::Matrix` class, or a class satisfying the `MockMatrix` interface.
 */
template<class Matrix_>
using WrappedAdjointWorkspace = typename get_adjoint_workspace<Matrix_>::type;

/**
 * @tparam Matrix_ A floating-point `Eigen::Matrix` class, or a class satisfying the `MockMatrix` interface.
 * @param matrix Instance of a matrix.
 * @return Workspace to use in `adjoint_multiply()`.
 */
template<class Matrix_>
WrappedAdjointWorkspace<Matrix_> wrapped_adjoint_workspace(const Matrix_& matrix) {
    if constexpr(std::is_same<WrappedAdjointWorkspace<Matrix_>, bool>::value) {
        return false;
    } else {
        return matrix.adjoint_workspace();
    }
}

/**
 * @cond
 */
namespace internal {

template<class Matrix_, typename = int>
struct is_eigen {
    static constexpr bool value = false;
};

template<class Matrix_>
struct is_eigen<Matrix_, decltype((void) std::declval<typename Matrix_::Index>(), 0)> {
    static constexpr bool value = true;
};

}
/**
 * @endcond
 */

/**
 * @tparam Matrix_ Class satisfying the `MockMatrix` interface, or a floating-point `EigenMatrix`.
 * @tparam Right_ A floating-point `Eigen::Vector` or equivalent expression.
 * @tparam EigenVector_ A floating-point `Eigen::Vector` class.
 *
 * @param[in] matrix Pointer to the wrapped matrix instance.
 * @param[in] rhs The right-hand side of the matrix product.
 * @param work Workspace for the matrix multiplication.
 * @param[out] out The output vector to store the matrix product.
 * This is filled with the product of this matrix and `rhs`.
 */
template<class Matrix_, class Right_, class EigenVector_>
void wrapped_multiply(const Matrix_& matrix, const Right_& rhs, WrappedWorkspace<Matrix_>& work, EigenVector_& out) {
    if constexpr(internal::is_eigen<Matrix_>::value) {
        out.noalias() = matrix * rhs;
    } else {
        matrix.multiply(rhs, work, out);
    }
}

/**
 * @tparam Matrix_ Class satisfying the `MockMatrix` interface, or a floating-point `EigenMatrix`.
 * @tparam Right_ A floating-point `Eigen::Vector` or equivalent expression.
 * @tparam EigenVector_ A floating-point `Eigen::Vector` class
 *
 * @param[in] matrix Pointer to the wrapped matrix instance.
 * @param[in] rhs The right-hand side of the matrix product.
 * @param work Workspace for the adjoint matrix multiplication.
 * @param[out] out The output vector to store the matrix product.
 * This is filled with the product of this matrix and `rhs`.
 */
template<class Matrix_, class Right_, class EigenVector_>
void wrapped_adjoint_multiply(const Matrix_& matrix, const Right_& rhs, WrappedAdjointWorkspace<Matrix_>& work, EigenVector_& out) {
    if constexpr(internal::is_eigen<Matrix_>::value) {
        out.noalias() = matrix.adjoint() * rhs;
    } else {
        matrix.adjoint_multiply(rhs, work, out);
    }
}

/**
 * @tparam EigenMatrix_ A floating-point `Eigen::Matrix`.
 * @tparam Matrix_ Class satisfying the `MockMatrix` interface, or a floating-point `Eigen::Matrix`.
 * @param[in] matrix Pointer to the wrapped matrix instance.
 * @return A dense **Eigen** matrix containing the realized contents of `mat`.
 */
template<class EigenMatrix_, class Matrix_>
EigenMatrix_ wrapped_realize(const Matrix_& matrix) {
    if constexpr(internal::is_eigen<Matrix_>::value) {
        return EigenMatrix_(matrix);
    } else {
        return matrix.template realize<EigenMatrix_>();
    }
}

/**
 * @brief Wrapper for a centered matrix.
 *
 * @tparam Matrix_ Class satisfying the `MockMatrix` interface, or a floating-point `Eigen::Matrix`.
 * @tparam EigenVector_ A floating-point `Eigen::Vector` class for the column centers and matrix-vector product.
 * 
 * This class computes the matrix-vector product after centering all columns in `Matrix_`, i.e., subtracting the mean of each column from the values of that column.
 * Naively doing such an operation would involve loss of sparsity, which we avoid by deferring the subtraction into the subspace defined by `rhs`.
 *
 * This class satisfies the `MockMatrix` interface and implements all of its methods/typedefs.
 */
template<class Matrix_, class EigenVector_>
class Centered {
public:
    /**
     * @param matrix Matrix to be column-centered.
     * @param center Vector of length equal to the number of columns of `matrix`,
     * containing the value to subtract from each column.
     */
    Centered(const Matrix_& matrix, const EigenVector_& center) : my_matrix(matrix), my_center(center) {}

    /**
     * @cond
     */
public:
    Eigen::Index rows() const { return my_matrix.rows(); }

    Eigen::Index cols() const { return my_matrix.cols(); }

public:
    struct Workspace {
        Workspace(WrappedWorkspace<Matrix_> i) : inner(std::move(i)) {}
        WrappedWorkspace<Matrix_> inner;
        EigenVector_ buffer;
    };

    Workspace workspace() const {
        return Workspace(wrapped_workspace(my_matrix));
    }

    struct AdjointWorkspace {
        AdjointWorkspace(WrappedAdjointWorkspace<Matrix_> i) : inner(std::move(i)) {}
        WrappedAdjointWorkspace<Matrix_> inner;
        EigenVector_ buffer;
    };

    AdjointWorkspace adjoint_workspace() const {
        return AdjointWorkspace(wrapped_adjoint_workspace(my_matrix));
    }

public:
    template<class Right_>
    void multiply(const Right_& rhs, Workspace& work, EigenVector_& out) const {
        const auto& realized_rhs = internal::realize_rhs(rhs, work.buffer);
        wrapped_multiply(my_matrix, realized_rhs, work.inner, out);
        auto beta = realized_rhs.dot(my_center);
        for (auto& o : out) {
            o -= beta;
        }
        return;
    }

    template<class Right_>
    void adjoint_multiply(const Right_& rhs, AdjointWorkspace& work, EigenVector_& out) const {
        const auto& realized_rhs = internal::realize_rhs(rhs, work.buffer);
        wrapped_adjoint_multiply(my_matrix, realized_rhs, work.inner, out);
        auto beta = realized_rhs.sum();
        out -= beta * (my_center);
        return;
    }

    template<class EigenMatrix_>
    Eigen::MatrixXd realize() const {
        auto output = wrapped_realize<EigenMatrix_>(my_matrix);
        output.array().rowwise() -= my_center.adjoint().array();
        return output;
    }
    /**
     * @endcond
     */

private:
    const Matrix_& my_matrix;
    const EigenVector_& my_center;
};

/**
 * @brief Wrapper for a scaled matrix.
 *
 * @param by_column_ Whether to scale the columns.
 * If `false`, scaling is applied to the rows instead.
 * @tparam Matrix_ Class satisfying the `MockMatrix` interface, or a floating-point `Eigen::Matrix`.
 * @tparam EigenVector_ A floating-point `Eigen::Vector` class for the scaling factors and matrix-vector product.
 * 
 * This class computes the matrix-vector product after scaling all rows or columns in `Matrix_`, i.e., multiplying or dividing the values of each row/column by some arbitrary value.
 * For example, we can use this to divide each column by the standard deviation to achieve unit variance in principal components analyses.
 * Naively doing such an operation would involve a copy of the matrix, which we avoid by deferring the scaling into the subspace defined by `rhs`.
 *
 * This class satisfies the `MockMatrix` interface and implements all of its methods/typedefs.
 */
template<bool column_, class Matrix_, class EigenVector_>
class Scaled {
public:
    /**
     * @param matrix Underlying matrix to be column-scaled (if `by_column_ = true`) or row-scaled (otherwise).
     * @param scale Vector of length equal to the number of columns (if `by_column_ = true`) or rows (otherwise) of `m`,
     * containing the scaling factor to divide (if `divide = true`) or multiply (otherwise) to each column/row.
     * @param divide Whether to divide by the supplied scaling factors.
     */
    Scaled(const Matrix_& matrix, const EigenVector_& scale, bool divide) : 
        my_matrix(matrix), my_scale(scale), my_divide(divide) {}

    /**
     * @cond
     */
public:
    Eigen::Index rows() const { return my_matrix.rows(); }

    Eigen::Index cols() const { return my_matrix.cols(); }

public:
    template<template<class> class Wrapper_>
    struct BufferedWorkspace {
        BufferedWorkspace(size_t n, Wrapper_<Matrix_> c) : buffer(n), child(std::move(c)) {}
        EigenVector_ buffer;
        Wrapper_<Matrix_> child;
    };

    typedef typename std::conditional<column_, BufferedWorkspace<WrappedWorkspace>, WrappedWorkspace<Matrix_> >::type Workspace;

    Workspace workspace() const {
        if constexpr(column_) {
            return BufferedWorkspace<WrappedWorkspace>(my_matrix.cols(), wrapped_workspace(my_matrix));
        } else {
            return wrapped_workspace(my_matrix);
        }
    }

    typedef typename std::conditional<column_, WrappedAdjointWorkspace<Matrix_>, BufferedWorkspace<WrappedAdjointWorkspace> >::type AdjointWorkspace;

    AdjointWorkspace adjoint_workspace() const {
        if constexpr(column_) {
            return wrapped_adjoint_workspace(my_matrix);
        } else {
            return BufferedWorkspace<WrappedAdjointWorkspace>(my_matrix.rows(), wrapped_adjoint_workspace(my_matrix));
        }
    }

public:
    template<class Right_>
    void multiply(const Right_& rhs, Workspace& work, EigenVector_& out) const {
        if constexpr(column_) {
            if (my_divide) {
                // We store the result here, because the underlying matrix's multiply()
                // might need to access rhs/scale multiple times, especially if it's
                // parallelized. Better to pay the cost of accessing a separate memory
                // space than computing the quotient repeatedly.
                work.buffer = rhs.cwiseQuotient(my_scale);
            } else {
                work.buffer = rhs.cwiseProduct(my_scale);
            }
            wrapped_multiply(my_matrix, work.buffer, work.child, out);

        } else {
            wrapped_multiply(my_matrix, rhs, work, out);
            if (my_divide) {
                out.array() /= my_scale.array();
            } else {
                out.array() *= my_scale.array();
            }
        }
    }

    template<class Right_>
    void adjoint_multiply(const Right_& rhs, AdjointWorkspace& work, EigenVector_& out) const {
        if constexpr(column_) {
            wrapped_adjoint_multiply(my_matrix, rhs, work, out);
            if (my_divide) {
                out.array() /= my_scale.array();
            } else {
                out.array() *= my_scale.array();
            }

        } else {
            if (my_divide) {
                work.buffer = rhs.cwiseQuotient(my_scale);
            } else {
                work.buffer = rhs.cwiseProduct(my_scale);
            }
            wrapped_adjoint_multiply(my_matrix, work.buffer, work.child, out);
        }
    }

    template<class EigenMatrix_>
    EigenMatrix_ realize() const {
        auto output = wrapped_realize<EigenMatrix_>(my_matrix);

        if constexpr(column_) {
            if (my_divide) {
                output.array().rowwise() /= my_scale.adjoint().array();
            } else {
                output.array().rowwise() *= my_scale.adjoint().array();
            }
        } else {
            if (my_divide) {
                output.array().colwise() /= my_scale.array();
            } else {
                output.array().colwise() *= my_scale.array();
            }
        }

        return output;
    }
    /**
     * @endcond
     */

private:
    const Matrix_& my_matrix;
    const EigenVector_& my_scale;
    bool my_divide;
};

/**
 * A convenient maker function to enable partial template deduction on the `Scaled` class.
 *
 * @tparam column_ Whether to scale the columns.
 * If `false`, scaling is applied to the rows instead.
 * @tparam Matrix_ Class satisfying the `MockMatrix` interface, or a floating-point `Eigen::Matrix`.
 * @tparam EigenVector_ A floating-point `Eigen::Vector` class for the scaling factors and matrix-vector product.
 *
 * @param matrix Underlying matrix to be column-scaled (if `by_column_ = true`) or row-scaled (otherwise).
 * @param scale Vector of length equal to the number of columns (if `by_column_ = true`) or rows (otherwise) of `m`,
 * containing the scaling factor to divide (if `divide = true`) or multiply (otherwise) to each column/row.
 * @param divide Whether to divide by the supplied scaling factors.
 *
 * @return A `Scaled` object.
 */
template<bool column_, class Matrix_, class EigenVector_>
Scaled<column_, Matrix_, EigenVector_> make_Scaled(const Matrix_& matrix, const EigenVector_& scale, bool divide) {
    return Scaled<column_, Matrix_, EigenVector_>(matrix, scale, divide);
}

}

#endif

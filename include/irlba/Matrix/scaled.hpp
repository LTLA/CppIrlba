#ifndef IRLBA_MATRIX_SCALED_HPP
#define IRLBA_MATRIX_SCALED_HPP

#include <memory>

#include "../utils.hpp"
#include "interface.hpp"

#include "Eigen/Dense"

/**
 * @file scaled.hpp
 * @brief Deferred scaling for a matrix.
 */

namespace irlba {

/**
 * @brief Workspace class for multiplying a `ScaledMatrix`.
 *
 * @tparam EigenVector_ A floating-point `Eigen::Vector` to be used as input/output of the multiplication.
 * @tparam Matrix_ Class of the matrix to be scaled, as referenced by the `ScaledMatrix`. 
 * @tparam Scale_ An **Eigen** vector class, to hold the scaling factors.
 *
 * Typically constructed by `ScaledMatrix::new_workspace()`.
 */
template<class EigenVector_, class Matrix_, class Scale_>
class ScaledWorkspace final : public Workspace<EigenVector_> {
public:
    /**
     * @cond
     */
    ScaledWorkspace(const Matrix_& matrix, const Scale_& scale, const bool column, const bool divide) : 
        my_work(matrix.new_known_workspace()),
        my_scale(scale),
        my_column(column),
        my_divide(divide)
    {}
    /**
     * @endcond
     */

private:
    I<decltype(std::declval<Matrix_>().new_known_workspace())> my_work;
    const Scale_& my_scale;
    bool my_column;
    bool my_divide;

    EigenVector_ my_buffer;

public:
    void multiply(const EigenVector_& right, EigenVector_& out) {
        if (my_column) {
            if (my_divide) {
                my_buffer = right.cwiseQuotient(my_scale);
            } else {
                my_buffer = right.cwiseProduct(my_scale);
            }
            my_work->multiply(my_buffer, out);

        } else {
            my_work->multiply(right, out);
            if (my_divide) {
                out.array() /= my_scale.array();
            } else {
                out.array() *= my_scale.array();
            }
        }
    }
};

/** 
 * @brief Workspace class for multiplying a transposed `ScaledMatrix`.
 *
 * @tparam EigenVector_ A floating-point `Eigen::Vector` to be used as input/output of the multiplication.
 * @tparam Matrix_ Class of the matrix to be scaled, as referenced by the `ScaledMatrix`. 
 * @tparam Scale_ An **Eigen** vector class, to hold the scaling factors.
 *
 * Typically constructed by `ScaledMatrix::new_adjoint_workspace()`.
 */
template<class EigenVector_, class Matrix_, class Scale_>
class ScaledAdjointWorkspace final : public AdjointWorkspace<EigenVector_> {
public:
    /**
     * @cond
     */
    ScaledAdjointWorkspace(const Matrix_& matrix, const Scale_& scale, const bool column, const bool divide) :
        my_work(matrix.new_known_adjoint_workspace()),
        my_scale(scale),
        my_column(column),
        my_divide(divide)
    {}
    /**
     * @endcond
     */

private:
    I<decltype(std::declval<Matrix_>().new_known_adjoint_workspace())> my_work;
    const Scale_& my_scale;
    bool my_column;
    bool my_divide;

    EigenVector_ my_buffer;

public:
    void multiply(const EigenVector_& right, EigenVector_& out) {
        if (my_column) {
            my_work->multiply(right, out);
            if (my_divide) {
                out.array() /= my_scale.array();
            } else {
                out.array() *= my_scale.array();
            }

        } else {
            if (my_divide) {
                my_buffer = right.cwiseQuotient(my_scale);
            } else {
                my_buffer = right.cwiseProduct(my_scale);
            }
            my_work->multiply(my_buffer, out);
        }
    }
};

/** 
 * @brief Workspace class for realizing a `ScaledMatrix`.
 *
 * @tparam EigenMatrix_ A dense floating-point `Eigen::Matrix` in which to store the realized matrix.
 * @tparam Matrix_ Class of the matrix to be centered, as referenced by the `ScaledMatrix`. 
 * @tparam Scale_ An **Eigen** vector class, to hold the scaling factors.
 *
 * Typically constructed by `ScaledMatrix::new_realize_workspace()`.
 */
template<class EigenMatrix_, class Matrix_, class Scale_>
class ScaledRealizeWorkspace final : public RealizeWorkspace<EigenMatrix_> {
public:
    /**
     * @cond
     */
    ScaledRealizeWorkspace(const Matrix_& matrix, const Scale_& scale, const bool column, const bool divide) :
        my_work(matrix.new_known_realize_workspace()),
        my_scale(scale),
        my_column(column),
        my_divide(divide)
    {}
    /**
     * @endcond
     */

private:
    I<decltype(std::declval<Matrix_>().new_known_realize_workspace())> my_work;
    const Scale_& my_scale;
    bool my_column;
    bool my_divide;

public:
    const EigenMatrix_& realize(EigenMatrix_& buffer) {
        my_work->realize_copy(buffer);

        if (my_column) {
            if (my_divide) {
                buffer.array().rowwise() /= my_scale.adjoint().array();
            } else {
                buffer.array().rowwise() *= my_scale.adjoint().array();
            }

        } else {
            if (my_divide) {
                buffer.array().colwise() /= my_scale.array();
            } else {
                buffer.array().colwise() *= my_scale.array();
            }
        }

        return buffer;
    }
};

/**
 * @brief Deferred scaling of a matrix.
 *
 * @tparam EigenVector_ A floating-point `Eigen::Vector` to be used as input/output of multiplication operations.
 * @tparam EigenMatrix_ A dense floating-point `Eigen::Matrix` in which to store the realized matrix.
 * Typically of the same scalar type as `EigenVector_`.
 * @tparam MatrixPointer_ Pointer to an instance of a class satisfying the `Matrix` interface.
 * This can be a smart or raw pointer.
 * @tparam ScalePointer_ Pointer to an instance of an **Eigen** vector class to hold the scaling facrors.
 * This can be a smart or raw pointer.
 * 
 * This class computes the matrix-vector product after scaling all rows or columns in the matrix, i.e., multiplying or dividing the values of each row/column by some arbitrary value.
 * For example, we can use this to divide each column by the standard deviation to achieve unit variance in principal components analyses.
 * Naively doing such an operation would involve a copy of the matrix, which we avoid by deferring the scaling into the subspace defined by `rhs`.
 */
template<
    class EigenVector_,
    class EigenMatrix_,
    class MatrixPointer_ = const Matrix<EigenVector_, EigenMatrix_>*,
    class ScalePointer_ = const EigenVector_*
>
class ScaledMatrix final : public Matrix<EigenVector_, EigenMatrix_> {
public:
    /**
     * @param matrix Pointer to a matrix to be column-scaled (if `column_ = true`) or row-scaled (otherwise).
     * @param scale Pointer to a vector of length equal to the number of columns (if `column_ = true`) or rows (otherwise) of `matrix`,
     * containing the scaling factor to divide (if `divide = true`) or multiply (otherwise) to each column/row.
     * @param column Whether to multiply/divide each column of `matrix` by the corresponding factor in `scale`.
     * If `false`, each row of `matrix` is scaled instead.
     * @param divide Whether to divide by the supplied scaling factors.
     * If `false`, multiplication is performed instead.
     */
    ScaledMatrix(MatrixPointer_ matrix, ScalePointer_ scale, bool column, bool divide) : 
        my_matrix(std::move(matrix)),
        my_scale(std::move(scale)),
        my_column(column),
        my_divide(divide)
    {}

private:
    MatrixPointer_ my_matrix;
    ScalePointer_ my_scale;
    bool my_column;
    bool my_divide;

public:
    Eigen::Index rows() const {
        return my_matrix->rows();
    }

    Eigen::Index cols() const {
        return my_matrix->cols();
    }

public:
    std::unique_ptr<Workspace<EigenVector_> > new_workspace() const {
        return new_known_workspace();
    }

    std::unique_ptr<AdjointWorkspace<EigenVector_> > new_adjoint_workspace() const {
        return new_known_adjoint_workspace();
    }

    std::unique_ptr<RealizeWorkspace<EigenMatrix_> > new_realize_workspace() const {
        return new_known_realize_workspace();
    }

public:
    /**
     * Overrides `Matrix::new_known_workspace()` to enable devirtualization.
     */
    std::unique_ptr<ScaledWorkspace<EigenVector_, I<decltype(*my_matrix)>, I<decltype(*my_scale)> > > new_known_workspace() const {
        return std::make_unique<ScaledWorkspace<EigenVector_, I<decltype(*my_matrix)>, I<decltype(*my_scale)> > >(*my_matrix, *my_scale, my_column, my_divide);
    }

    /**
     * Overrides `Matrix::new_known_adjoint_workspace()` to enable devirtualization.
     */
    std::unique_ptr<ScaledAdjointWorkspace<EigenVector_, I<decltype(*my_matrix)>, I<decltype(*my_scale)> > > new_known_adjoint_workspace() const {
        return std::make_unique<ScaledAdjointWorkspace<EigenVector_, I<decltype(*my_matrix)>, I<decltype(*my_scale)> > >(*my_matrix, *my_scale, my_column, my_divide);
    }

    /**
     * Overrides `Matrix::new_known_realize_workspace()` to enable devirtualization.
     */
    std::unique_ptr<ScaledRealizeWorkspace<EigenMatrix_, I<decltype(*my_matrix)>, I<decltype(*my_scale)> > > new_known_realize_workspace() const {
        return std::make_unique<ScaledRealizeWorkspace<EigenMatrix_, I<decltype(*my_matrix)>, I<decltype(*my_scale)> > >(*my_matrix, *my_scale, my_column, my_divide);
    }
};

}

#endif

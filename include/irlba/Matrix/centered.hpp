#ifndef IRLBA_MATRIX_CENTERED_HPP
#define IRLBA_MATRIX_CENTERED_HPP

#include <memory>

#include "../utils.hpp"
#include "interface.hpp"

#include "Eigen/Dense"

/**
 * @file centered.hpp
 * @brief Deferred centering for a matrix.
 */

namespace irlba {

/**
 * @brief Workspace class for multiplying a `CenteredMatrix`.
 *
 * @tparam EigenVector_ A floating-point `Eigen::Vector` to be used as input/output of the multiplication. 
 * @tparam Matrix_ Class of the matrix to be centered, as referenced by the `CenteredMatrix`. 
 * @tparam Center_ An **Eigen** vector class, to hold the column centers.
 *
 * Typically constructed by `CenteredMatrix::new_workspace()`.
 */
template<class EigenVector_, class Matrix_, class Center_>
class CenteredWorkspace final : public Workspace<EigenVector_> {
public:
    /**
     * @cond
     */
    CenteredWorkspace(const Matrix_& matrix, const Center_& center) : 
        my_work(matrix.new_known_workspace()),
        my_center(center)
    {}
    /**
     * @endcond
     */

private:
    I<decltype(std::declval<Matrix_>().new_known_workspace())> my_work;
    const Center_& my_center;

public:
    void multiply(const EigenVector_& right, EigenVector_& out) {
        my_work->multiply(right, out);
        const auto beta = right.dot(my_center);
        for (auto& o : out) {
            o -= beta;
        }
    }
};

/** 
 * @brief Workspace class for multiplying a transposed `CenteredMatrix`.
 *
 * @tparam EigenVector_ A floating-point `Eigen::Vector` to be used as input/output of the multiplication.
 * @tparam Matrix_ Class of the matrix to be centered, as referenced by the `CenteredMatrix`. 
 * @tparam Center_ An **Eigen** vector class, to hold the column centers.
 *
 * Typically constructed by `CenteredMatrix::new_adjoint_workspace()`.
 */
template<class EigenVector_, class Matrix_, class Center_>
class CenteredAdjointWorkspace final : public AdjointWorkspace<EigenVector_> {
public:
    /**
     * @cond
     */
    CenteredAdjointWorkspace(const Matrix_& matrix, const Center_& center) :
        my_work(matrix.new_known_adjoint_workspace()),
        my_center(center)
    {}
    /**
     * @endcond
     */

private:
    I<decltype(std::declval<Matrix_>().new_known_adjoint_workspace())> my_work;
    const Center_& my_center;

public:
    void multiply(const EigenVector_& right, EigenVector_& out) {
        my_work->multiply(right, out);
        const auto beta = right.sum();
        out -= beta * my_center;
    }
};

/** 
 * @brief Workspace class for realizing a `CenteredMatrix`.
 *
 * @tparam EigenMatrix_ A dense floating-point `Eigen::Matrix` in which to store the realized matrix.
 * @tparam Matrix_ Class of the matrix to be centered, as referenced by the `CenteredMatrix`. 
 * @tparam Center_ An **Eigen** vector class, to hold the column centers.
 *
 * Typically constructed by `CenteredMatrix::new_realize_workspace()`.
 */
template<class EigenMatrix_, class Matrix_, class Center_>
class CenteredRealizeWorkspace final : public RealizeWorkspace<EigenMatrix_> {
public:
    /**
     * @cond
     */
    CenteredRealizeWorkspace(const Matrix_& matrix, const Center_& center) :
        my_work(matrix.new_known_realize_workspace()),
        my_center(center)
    {}
    /**
     * @endcond
     */

private:
    I<decltype(std::declval<Matrix_>().new_known_realize_workspace())> my_work;
    const Center_& my_center;

public:
    const EigenMatrix_& realize(EigenMatrix_& buffer) {
        my_work->realize_copy(buffer);
        buffer.array().rowwise() -= my_center.adjoint().array();
        return buffer;
    }
};

/**
 * @brief Deferred centering of a matrix.
 *
 * @tparam EigenVector_ A floating-point `Eigen::Vector` to be used as input/output of multiplication operations.
 * @tparam EigenMatrix_ A dense floating-point `Eigen::Matrix` in which to store the realized matrix.
 * Typically of the same scalar type as `EigenVector_`.
 * @tparam MatrixPointer_ Pointer to an instance of a class satisfying the `Matrix` interface.
 * This can be a smart or raw pointer.
 * @tparam CenterPointer_ Pointer to an instance of an **Eigen** vector class to hold the column centers.
 * This can be a smart or raw pointer.
 * 
 * This class computes the matrix-vector product after centering all columns in the matrix, i.e., subtracting the mean of each column from the values of that column.
 * Naively doing such an operation would involve loss of sparsity, which we avoid by deferring the subtraction into the subspace defined by `right`.
 */
template<
    class EigenVector_,
    class EigenMatrix_,
    class MatrixPointer_ = const Matrix<EigenVector_, EigenMatrix_>*,
    class CenterPointer_ = const EigenVector_* 
>
class CenteredMatrix final : public Matrix<EigenVector_, EigenMatrix_> {
public:
    /**
     * @param matrix Pointer to a matrix to be column-centered.
     * @param center Pointer to a vector of length equal to the number of columns of `matrix`,
     * containing the value to subtract from each column.
     */
    CenteredMatrix(MatrixPointer_ matrix, CenterPointer_ center) :
        my_matrix(std::move(matrix)),
        my_center(std::move(center))
    {}

private:
    MatrixPointer_ my_matrix;
    CenterPointer_ my_center;

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
    std::unique_ptr<CenteredWorkspace<EigenVector_, I<decltype(*my_matrix)>, I<decltype(*my_center)> > > new_known_workspace() const {
        return std::make_unique<CenteredWorkspace<EigenVector_, I<decltype(*my_matrix)>, I<decltype(*my_center)> > >(*my_matrix, *my_center);
    }

    /**
     * Overrides `Matrix::new_known_adjoint_workspace()` to enable devirtualization.
     */
    std::unique_ptr<CenteredAdjointWorkspace<EigenVector_, I<decltype(*my_matrix)>, I<decltype(*my_center)> > > new_known_adjoint_workspace() const {
        return std::make_unique<CenteredAdjointWorkspace<EigenVector_, I<decltype(*my_matrix)>, I<decltype(*my_center)> > >(*my_matrix, *my_center);
    }

    /**
     * Overrides `Matrix::new_known_realize_workspace()` to enable devirtualization.
     */
    std::unique_ptr<CenteredRealizeWorkspace<EigenMatrix_, I<decltype(*my_matrix)>, I<decltype(*my_center)> > > new_known_realize_workspace() const {
        return std::make_unique<CenteredRealizeWorkspace<EigenMatrix_, I<decltype(*my_matrix)>, I<decltype(*my_center)> > >(*my_matrix, *my_center);
    }
};

}

#endif

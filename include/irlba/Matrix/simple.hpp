#ifndef IRLBA_MATRIX_SIMPLE_HPP
#define IRLBA_MATRIX_SIMPLE_HPP

#include <memory>
#include <type_traits>

#include "../utils.hpp"
#include "interface.hpp"

#include "Eigen/Dense"

/**
 * @file simple.hpp
 * @brief Simple wrapper around an **Eigen** matrix.
 */

namespace irlba {

/**
 * @brief Workspace class for multiplying a `SimpleMatrix`.
 *
 * @tparam EigenVector_ A floating-point `Eigen::Vector` to be used as input/output of the multiplication.
 * @tparam Simple_ The underlying matrix referenced by the `SimpleMatrix`. 
 *
 * Typically constructed by `SimpleMatrix::new_workspace()`.
 */
template<class EigenVector_, class Simple_>
class SimpleWorkspace final : public Workspace<EigenVector_> {
public:
    /**
     * @cond
     */
    SimpleWorkspace(const Simple_& matrix) : my_matrix(matrix) {}
    /**
     * @endcond
     */

private:
    const Simple_& my_matrix;

public:
    void multiply(const EigenVector_& right, EigenVector_& output) {
        output.noalias() = my_matrix * right;
    }
};

/**
 * @brief Workspace class for multiplying a transposed `SimpleMatrix`.
 *
 * @tparam EigenVector_ A floating-point `Eigen::Vector` to be used as input/output of the multiplication.
 * @tparam Simple_ The underlying matrix referenced by the `SimpleMatrix`. 
 *
 * Typically constructed by `SimpleMatrix::new_adjoint_workspace()`.
 */
template<class EigenVector_, class Simple_>
class SimpleAdjointWorkspace final : public AdjointWorkspace<EigenVector_> {
public:
    /**
     * @cond
     */
    SimpleAdjointWorkspace(const Simple_& matrix) : my_matrix(matrix) {}
    /**
     * @endcond
     */

private:
    const Simple_& my_matrix;

public:
    void multiply(const EigenVector_& right, EigenVector_& output) {
        output.noalias() = my_matrix.adjoint() * right;
    }
};

/**
 * @brief Workspace class for realizing a `SimpleMatrix`.
 *
 * @tparam EigenMatrix_ A dense floating-point `Eigen::Matrix` in which to store the realized matrix.
 * @tparam Simple_ The underlying matrix referenced by the `SimpleMatrix`. 
 *
 * Typically constructed by `SimpleMatrix::new_realize_workspace()`.
 */
template<class EigenMatrix_, class Simple_>
class SimpleRealizeWorkspace final : public RealizeWorkspace<EigenMatrix_> {
public:
    /**
     * @cond
     */
    SimpleRealizeWorkspace(const Simple_& matrix) : my_matrix(matrix) {}
    /**
     * @endcond
     */

private:
    const Simple_& my_matrix;
    static constexpr bool is_same = std::is_same<EigenMatrix_, I<Simple_> >::value;
    typename std::conditional<is_same, bool, EigenMatrix_>::type buffer;

public:
    const EigenMatrix_& realize(EigenMatrix_& buffer) {
        if constexpr(is_same) {
            return my_matrix;
        } else {
            buffer = my_matrix;
            return buffer;
        }
    }
};

/**
 * @brief A `Matrix`-compatible wrapper around a "simple" matrix.
 *
 * @tparam EigenVector_ A floating-point `Eigen::Vector` to be used as input/output of multiplication operations.
 * @tparam EigenMatrix_ A dense floating-point `Eigen::Matrix` in which to store the realized matrix.
 * Typically of the same scalar type as `EigenVector_`.
 * @tparam SimplePointer_ Pointer to an instance of a simple matrix class. 
 * This may be a raw or smart pointer.
 *
 * A simple matrix is one that implements right-side multiplication by an `EigenVector_`,
 * right-side multiplication on the result of `adjoint()`,
 * and copy assignment to an `EigenMatrix_`.
 * This is most typically an instance of an **Eigen** matrix class, though the exact class need not be the same as `EigenMatrix_`.
 */
template<class EigenVector_, class EigenMatrix_, class SimplePointer_>
class SimpleMatrix final : public Matrix<EigenVector_, EigenMatrix_> {
public:
    /**
     * @param matrix Pointer to a simple matrix.
     */
    SimpleMatrix(SimplePointer_ matrix) : my_matrix(std::move(matrix)) {}

private:
    SimplePointer_ my_matrix;

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
    std::unique_ptr<SimpleWorkspace<EigenVector_, I<decltype(*my_matrix)> > > new_known_workspace() const {
        return std::make_unique<SimpleWorkspace<EigenVector_, I<decltype(*my_matrix)> > >(*my_matrix);
    }

    /**
     * Overrides `Matrix::new_known_adjoint_workspace()` to enable devirtualization.
     */
    std::unique_ptr<SimpleAdjointWorkspace<EigenVector_, I<decltype(*my_matrix)> > > new_known_adjoint_workspace() const {
        return std::make_unique<SimpleAdjointWorkspace<EigenVector_, I<decltype(*my_matrix)> > >(*my_matrix);
    }

    /**
     * Overrides `Matrix::new_known_realize_workspace()` to enable devirtualization.
     */
    std::unique_ptr<SimpleRealizeWorkspace<EigenMatrix_, I<decltype(*my_matrix)> > > new_known_realize_workspace() const {
        return std::make_unique<SimpleRealizeWorkspace<EigenMatrix_, I<decltype(*my_matrix)> > >(*my_matrix);
    }
};

}

#endif

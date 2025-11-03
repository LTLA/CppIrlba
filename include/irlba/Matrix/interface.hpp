#ifndef IRLBA_MATRIX_INTERFACE_HPP
#define IRLBA_MATRIX_INTERFACE_HPP

#include <memory>

#include "Eigen/Dense"

/**
 * @file interface.hpp
 * @brief Interfaces for matrix inputs.
 */

namespace irlba {

/**
 * @brief Workspace class for multiplying a `Matrix`.
 *
 * @tparam EigenVector_ A floating-point `Eigen::Vector`.
 *
 * Typically constructed by `Matrix::new_workspace()`.
 * The lifetime of this object should not exceed that of the `Matrix` instance used to construct it.
 */
template<class EigenVector_>
class Workspace {
public:
    /**
     * @cond
     */
    Workspace() = default;
    Workspace(Workspace&&) = default;
    Workspace(const Workspace&) = default;
    Workspace& operator=(Workspace&&) = default;
    Workspace& operator=(const Workspace&) = default;
    virtual ~Workspace() {}
    /**
     * @endcond
     */

    /**
     * @param[in] right The right-hand side of the matrix product.
     * @param[out] output The output vector to store the matrix product.
     * This is filled with the product of this matrix and `rhs`.
     *
     * This method will be called without any explicit template arguments, 
     * so implementations do not need to use the same number/order of template parameters. 
     * `EigenVector_` may also be a template parameter of the class rather than the method,
     * depending on what is most convenient for defining the associated `Workspace`.
     */
    virtual void multiply(const EigenVector_& right, EigenVector_& output) = 0;
};

/**
 * @brief Workspace class for multiplying a transposed `Matrix`.
 *
 * @tparam EigenVector_ A floating-point `Eigen::Vector`.
 *
 * Typically constructed by `Matrix::new_adjoint_workspace()`.
 * The lifetime of this object should not exceed that of the `Matrix` instance used to construct it.
 */
template<class EigenVector_>
class AdjointWorkspace {
public:
    /**
     * @cond
     */
    AdjointWorkspace() = default;
    AdjointWorkspace(AdjointWorkspace&&) = default;
    AdjointWorkspace(const AdjointWorkspace&) = default;
    AdjointWorkspace& operator=(AdjointWorkspace&&) = default;
    AdjointWorkspace& operator=(const AdjointWorkspace&) = default;
    virtual ~AdjointWorkspace() {}
    /**
     * @endcond
     */

    /**
     *
     * @param[in] right The right-hand side of the matrix product.
     * @param[out] output The output vector to store the matrix product.
     * This is filled with the product of the transpose of this matrix and `rhs`.
     *
     * This method will be called without any explicit template arguments, 
     * so implementations do not need to use the same number/order of template parameters. 
     * `EigenVector_` may also be a template parameter of the class rather than the method.
     * depending on what is most convenient for defining the associated `AdjointWorkspace`.
     */
    virtual void multiply(const EigenVector_& right, EigenVector_& output) = 0;
};

/**
 * @brief Workspace class for realizing a `Matrix`.
 *
 * @tparam EigenMatrix_ A floating-point `Eigen::Matrix`.
 *
 * Typically constructed by `Matrix::new_realize_workspace()`.
 * The lifetime of this object should not exceed that of the `Matrix` instance used to construct it.
 */
template<class EigenMatrix_>
class RealizeWorkspace { 
public:
    /**
     * @cond
     */
    RealizeWorkspace() = default;
    RealizeWorkspace(RealizeWorkspace&&) = default;
    RealizeWorkspace(const RealizeWorkspace&) = default;
    RealizeWorkspace& operator=(RealizeWorkspace&&) = default;
    RealizeWorkspace& operator=(const RealizeWorkspace&) = default;
    virtual ~RealizeWorkspace() {}
    /**
     * @endcond
     */

    /**
     * @param buffer Buffer in which to optionally store the realized matrix.
     * @return Reference to a realized matrix.
     * This may refer to `buffer` or some other object.
     */
    virtual const EigenMatrix_& realize(EigenMatrix_& buffer) = 0;

    /**
     * @param[out] buffer Buffer in which to store the realized matrix.
     * Unlike `realize()`, this is guaranteed to contain the contents of the realized matrix.
     */
    void realize_copy(EigenMatrix_& buffer) {
        const auto& out = realize(buffer);
        if (&out != &buffer) {
            buffer = out;
        }
    }
};

/**
 * @brief Interface for a matrix to use in `compute()`.
 *
 * @tparam EigenVector_ A floating-point `Eigen::Vector`.
 * @tparam EigenMatrix_ A floating-point `Eigen::Matrix`.
 *
 * Defines an time interface for a matrix input to IRLBA, supporting matrix-vector multiplication and realization into an `EigenMatrix_`.
 */
template<class EigenVector_, class EigenMatrix_>
class Matrix {
public:
    /**
     * @cond
     */
    Matrix() = default;
    Matrix(Matrix&&) = default;
    Matrix(const Matrix&) = default;
    Matrix& operator=(Matrix&&) = default;
    Matrix& operator=(const Matrix&) = default;
    virtual ~Matrix() {}
    /**
     * @endcond
     */

public:
    /**
     * @return Number of rows in the matrix.
     */
    virtual Eigen::Index rows() const = 0;

    /**
     * @return Number of columns in the matrix.
     */
    virtual Eigen::Index cols() const = 0;

public:
    /**
     * @return Pointer to a new workspace for matrix multiplication.
     * The lifetime of this object should not exceed that of its parent `Matrix`.
     */
    virtual std::unique_ptr<Workspace<EigenVector_> > new_workspace() const = 0;

    /**
     * @return Pointer to a new workspace for adjoint matrix multiplication.
     * The lifetime of this object should not exceed that of its parent `Matrix`.
     */
    virtual std::unique_ptr<AdjointWorkspace<EigenVector_> > new_adjoint_workspace() const = 0;

    /**
     * @return Pointer to a new workspace for matrix realization.
     * The lifetime of this object should not exceed that of its parent `Matrix`.
     */
    virtual std::unique_ptr<RealizeWorkspace<EigenMatrix_> > new_realize_workspace() const = 0;

public:
    /**
     * @return A new workspace for matrix multiplication.
     * The lifetime of this object should not exceed that of its parent `Matrix`.
     *
     * Subclasses may override this method to return a pointer to a specific `Workspace` subclass.
     * This is used for devirtualization in `compute()`. 
     * If no override is provided, `new_workspace()` is called instead.
     */
    std::unique_ptr<Workspace<EigenVector_> > new_known_workspace() const {
        return new_workspace();
    }

    /**
     * @return A new workspace for adjoint matrix multiplication.
     * The lifetime of this object should not exceed that of its parent `Matrix`.
     *
     * Subclasses may override this method to return a pointer to a specific `AdjointWorkspace` subclass.
     * This is used for devirtualization in `compute()`. 
     * If no override is provided, `new_adjoint_workspace()` is called instead.
     */
    std::unique_ptr<AdjointWorkspace<EigenVector_> > new_known_adjoint_workspace() const {
        return new_adjoint_workspace();
    }

    /**
     * @return A new workspace for matrix realization.
     * The lifetime of this object should not exceed that of its parent `Matrix`.
     *
     * Subclasses may override this method to return a pointer to a specific `RealizeWorkspace` subclass.
     * This is used for devirtualization in `compute()`. 
     * If no override is provided, `new_realize_workspace()` is called instead.
     */
    std::unique_ptr<RealizeWorkspace<EigenMatrix_> > new_known_realize_workspace() const {
        return new_realize_workspace();
    }
};

}

#endif

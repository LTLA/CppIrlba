#ifndef IRLBA_MOCK_MATRIX_HPP
#define IRLBA_MOCK_MATRIX_HPP

#include "Eigen/Dense"

/**
 * @file MockMatrix.hpp
 * @brief Compile-time interface for matrix inputs.
 */

namespace irlba {

/**
 * @brief Interface for a matrix to use in `irlba::compute()`.
 *
 * Defines a compile-time interface for a matrix input to IRLBA.
 * This should mainly provide methods for multiplication with a vector.
 * Developers can also define wrappers to perform the matrix-vector multiplication after some operation like centering or scaling.
 * The idea is to compute the product without actually modifying the underlying matrix, which is important when the modification results in an unnecessary copy and/or loss of sparsity.
 * We achieve this by deferring the modification into the subspace defined by vector.
 */
class MockMatrix {
public:
    /**
     * @cond
     */
    MockMatrix(Eigen::MatrixXd x) : my_x(std::move(x)) {}
    /**
     * @endcond
     */

private:
    Eigen::MatrixXd my_x;

public:

    /**
     * @return Number of rows in the matrix.
     */
    Eigen::Index rows() const { return my_x.rows(); }

    /**
     * @return Number of columns in the matrix.
     */
    Eigen::Index cols() const { return my_x.cols(); }

public:
    /**
     * Workspace class, used to allocate space for intermediate results across multiple calls to `multiply()`.
     */
    struct Workspace {};

    /**
     * @return Workspace for use in `multiply()`.
     */
    Workspace workspace() const {
        return Workspace();
    }

    /**
     * Workspace class, used to allocate space for intermediate results across multiple calls to `adjoint_multiply()`.
     */
    struct AdjointWorkspace {};

    /**
     * @return Workspace for use in `adjoint_multiply()`.
     */
    AdjointWorkspace adjoint_workspace() const {
        return AdjointWorkspace();
    }

public:
    /**
     * @tparam Right_ A floating-point `Eigen::Vector` or equivalent expression.
     * @tparam EigenVector_ A floating-point `Eigen::Vector`.
     *
     * @param[in] rhs The right-hand side of the matrix product.
     * @param work The return value of `workspace()`.
     * This can be reused across multiple `multiply()` calls.
     * @param[out] out The output vector to store the matrix product.
     * This is filled with the product of this matrix and `rhs`.
     */
    template<class Right_, class EigenVector_>
    void multiply(const Right_& rhs, Workspace& work, EigenVector_& out) const {
        out.noalias() = my_x * rhs; 
    }

    /**
     * @tparam Right_ A floating-point `Eigen::Vector` or equivalent expression.
     * @tparam EigenVector_ A floating-point `Eigen::Vector`.
     *
     * @param[in] rhs The right-hand side of the matrix product.
     * @param work The return value of `adjoint_workspace()`.
     * This can be reused across multiple `adjoint_multiply()` calls.
     * @param[out] out The output vector to store the matrix product.
     * This is filled with the product of the transpose of this matrix and `rhs`.
     */
    template<class Right_, class EigenVector2_>
    void adjoint_multiply(const Right_& rhs, AdjointWorkspace& work, EigenVector2_& out) const {
        out.noalias() = my_x.adjoint() * rhs;
    }

    /**
     * @tparam EigenMatrix_ A floating-point `Eigen::Matrix`.
     * @return A realized version of the centered matrix,
     * where the centering has been explicitly applied.
     */
    template<class EigenMatrix_ = Eigen::MatrixXd>
    EigenMatrix_ realize() const {
        return wrapped_realize<EigenMatrix_>(mat);
    }
};

}

#endif

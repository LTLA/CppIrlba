#ifndef IRLBA_LANCZOS_HPP
#define IRLBA_LANCZOS_HPP

#include "Eigen/Dense"
#include "utils.hpp"
#include "wrappers.hpp"
#include <cmath>
#include <limits>

/**
 * @file lanczos.hpp
 *
 * @brief Perform the Lanczos bidiagonalization iterations.
 */

namespace irlba {

/**
 * @brief Perform Lanczos bidiagonalization on an input matrix.
 */
template<typename MatrixType = Eigen::MatrixXd>
class LanczosBidiagonalization {
public:
    using Scalar = typename MatrixType::Scalar;
    using VectorType = Eigen::Vector<Scalar, MatrixType::RowsAtCompileTime>;
    using RealVectorType = Eigen::Vector<typename Eigen::NumTraits<Scalar>::Real, MatrixType::RowsAtCompileTime>;
    struct Defaults {
        /**
         * See `set_epsilon()` for details.
         */
        static constexpr double epsilon = -1;
    };
public:
    /**
     * Set the tolerance to use to define invariant subspaces.
     * This is used as the lower bound for the L2 norm for the subspace vectors;
     * below this bound, vectors are treated as all-zero and are instead filled with random draws from a normal distribution.
     *
     * @param e A positive number defining the tolerance.
     * If negative, we instead use the machine epsilon to the power of 0.8 (the same value as the **irlba** R package).
     *
     * @return A reference to the `LanczosBidiagonalization` instance.
     */
    LanczosBidiagonalization& set_epsilon(double e = Defaults::epsilon) {
        epsilon = e;
        return *this;
    }

public:
    /**
     * @brief Intermediate data structures to avoid repeated allocations.
     */
    struct Intermediates {
        /**
         * @tparam M Matrix class, most typically from the **Eigen** library.
         *
         * @param mat Instance of a matrix class `M`.
         */
        template<class M>
        Intermediates(const M& mat) : F(mat.cols()), W_next(mat.rows()), orthog_tmp(mat.cols()) {}

        /**
         * Obtain the residual vector, see algorithm 2.1 of Baglama and Reichel (2005).
         *
         * @return Vector of residuals of length equal to the number of columns of `mat` in `run()`.
         */
        const VectorType& residuals() const {
            return F;
        }

        /**
         * @cond
         */
        VectorType F; 
        VectorType W_next;
        VectorType orthog_tmp;
        /**
         * @endcond
         */
    };

    template<class M>
    Intermediates initialize(const M& mat) {
        return Intermediates(mat);
    }

public:
    /**
     * Perform the Lanczos bidiagonalization on an input matrix, optionally with scaling and centering.
     * This implements Algorithm 2.1 described by Baglama and Reichel (2005).
     * Support is provided for centering and scaling without modifying `mat`.
     * Protection against invariant subspaces is also implemented.
     *
     * @tparam M Matrix class, most typically from the **Eigen** library.
     * See the `Irlba` documentation for a detailed description of the expected methods.
     * @tparam Engine A functor that, when called with no arguments, returns a random integer from a discrete uniform distribution.
     *
     * @param mat Input matrix.
     * @param W Output matrix with number of rows equal to `mat.rows()`.
     * The size of the working subspace is defined from the number of columns.
     * The first `start` columns should contain orthonormal column vectors with non-zero L2 norms.
     * @param V Matrix with number of rows equal to `mat.cols()` and number of columns equal to `W.cols()`.
     * The first `start + 1` columns should contain orthonormal column vectors with non-zero L2 norms.
     * @param B Square matrix with number of rows and columns equal to the size of the working subspace.
     * Number of values is defined by `set_number()`.
     * @param eng An instance of a random number `Engine`.
     * @param inter Collection of intermediate data structures generated by calling `initialize()` on `mat`.
     * @param start The dimension from which to start the bidiagonalization.
     *
     * @return
     * `W` is filled with orthonormal vectors, as is `V`.
     * `B` is filled with upper diagonal entries.
     */
    template<class M, class Engine>
    void run(
        const M& mat, 
        MatrixType& W, 
        MatrixType& V, 
        MatrixType& B, 
        Engine& eng, 
        Intermediates& inter, 
        int start = 0) 
    {
        const double eps = (epsilon < 0 ? std::pow(std::numeric_limits<double>::epsilon(), 0.8) : epsilon);

        int work = W.cols();
        auto& F = inter.F;
        auto& W_next = inter.W_next;
        auto& otmp = inter.orthog_tmp;

        F = V.col(start);
        if constexpr(has_multiply_method<M, VectorType>::value) {
            W_next.noalias() = mat * F;
        } else {
            mat.multiply(F, W_next);
        }

        // If start = 0, we assume that it's already normalized, see argument description for 'V'.
        if (start) {
            orthogonalize_vector(W, W_next, start, otmp);
        }

        double S = W_next.norm();
        if (S < eps) {
            throw -4;
        }
        W_next /= S;
        W.col(start) = W_next;

        // The Lanczos iterations themselves.
        for (int j = start; j < work; ++j) {
            if constexpr(has_adjoint_multiply_method<M, MatrixType>::value) {
                F.noalias() = mat.adjoint() * W.col(j);
            } else {
                mat.adjoint_multiply(W.col(j), F);
            }

            F -= S * V.col(j); // equivalent to daxpy.
            orthogonalize_vector(V, F, j + 1, otmp);

            if (j + 1 < work) {
                double R_F = F.norm();

                if (R_F < eps) {
                    fill_with_random_normals(F, eng);
                    orthogonalize_vector(V, F, j + 1, otmp);
                    R_F = F.norm();
                    F /= R_F;
                    R_F = 0;
                } else {
                    F /= R_F;
                }

                V.col(j + 1) = F;

                B(j, j) = S;
                B(j, j + 1) = R_F;

                if constexpr(has_multiply_method<M, VectorType>::value) {
                    W_next.noalias() = mat * F;
                } else {
                    mat.multiply(F, W_next);
                }

                // Full re-orthogonalization, using the left-most 'j +  1' columns of W.
                // Recall that W_next will be the 'j + 2'-th column, i.e., W.col(j + 1) in
                // 0-indexed terms, so we want to orthogonalize to all previous columns.
                orthogonalize_vector(W, W_next, j + 1, otmp);

                S = W_next.norm();
                if (S < eps) {
                    fill_with_random_normals(W_next, eng);
                    orthogonalize_vector(W, W_next, j + 1, otmp);
                    S = W_next.norm();
                    W_next /= S;
                    S = 0;
                } else {
                    W_next /= S;
                }

                W.col(j + 1) = W_next;
            } else {
                B(j, j) = S;
            }
        }

        return;
    }

private:
    double epsilon = Defaults::epsilon;
};

}

#endif

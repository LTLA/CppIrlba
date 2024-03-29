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
class LanczosBidiagonalization {
public:
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
     * @tparam M Some kind of matrix class, either from the **Eigen** library or one of **irlba**'s wrappers.
     *
     * @brief Intermediate data structures to avoid repeated allocations on `run()`. 
     */
    template<class M>
    struct Intermediates {
        /**
         * @param mat Instance of a matrix class `M`.
         */
        Intermediates(const M& mat) : 
            F(mat.cols()), 
            W_next(mat.rows()), 
            orthog_tmp(mat.cols()), 
            work(wrapped_workspace(&mat)),
            awork(wrapped_adjoint_workspace(&mat)) 
        {}

        /**
         * Obtain the residual vector, see algorithm 2.1 of Baglama and Reichel (2005).
         *
         * @return Vector of residuals of length equal to the number of columns of `mat` in `run()`.
         */
        const Eigen::VectorXd& residuals() const {
            return F;
        }

        /**
         * @cond
         */
        Eigen::VectorXd F; 
        Eigen::VectorXd W_next;
        Eigen::VectorXd orthog_tmp;
        WrappedWorkspace<M> work;
        WrappedAdjointWorkspace<M> awork;
        /**
         * @endcond
         */
    };

    /**
     * @tparam M Some matrix class, either from the **Eigen** library or one of **irlba**'s wrappers.
     * @return An `Intermediates` object for subsequent calls to `run()` on `mat`.
     */
    template<class M>
    Intermediates<M> initialize(const M& mat) const {
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
     * @param[in, out] W Output matrix with number of rows equal to `mat.rows()`.
     * The size of the working subspace is defined from the number of columns.
     * The first `start` columns should contain orthonormal column vectors with non-zero L2 norms.
     * On output, the rest of `W` is filled with orthonormal vectors.
     * @param[in, out] V Matrix with number of rows equal to `mat.cols()` and number of columns equal to `W.cols()`.
     * The first `start + 1` columns should contain orthonormal column vectors with non-zero L2 norms.
     * On output, the rest of `V` is filled with orthonormal vectors.
     * @param[in, out] B Square matrix with number of rows and columns equal to the size of the working subspace.
     * Number of values is defined by `set_number()`.
     * On output, `B` is filled with upper diagonal entries, starting from the `start`-th row/column.
     * @param eng An instance of a random number `Engine`.
     * @param inter Collection of intermediate data structures generated by calling `initialize()` on `mat`.
     * @param start The dimension from which to start the bidiagonalization.
     */
    template<class M, class Engine>
    void run(
        const M& mat, 
        Eigen::MatrixXd& W, 
        Eigen::MatrixXd& V, 
        Eigen::MatrixXd& B, 
        Engine& eng, 
        Intermediates<M>& inter, 
        int start = 0) 
    const {
        const double eps = (epsilon < 0 ? std::pow(std::numeric_limits<double>::epsilon(), 0.8) : epsilon);

        int work = W.cols();
        auto& F = inter.F;
        auto& W_next = inter.W_next;
        auto& otmp = inter.orthog_tmp;

        F = V.col(start);
        wrapped_multiply(&mat, F, inter.work, W_next); // i.e., W_next = mat * F;

        // If start = 0, there's nothing to orthogonalize against.
        if (start) {
            orthogonalize_vector(W, W_next, start, otmp);
        }

        double S = W_next.norm();
        if (S < eps) {
            throw std::runtime_error("starting vector near the null space of the input matrix");
        }
        W_next /= S;
        W.col(start) = W_next;

        // The Lanczos iterations themselves.
        for (int j = start; j < work; ++j) {
            wrapped_adjoint_multiply(&mat, W.col(j), inter.awork, F); // i.e., F = mat.adjoint() * W.col(j);

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

                wrapped_multiply(&mat, F, inter.work, W_next); // i.e., W_next = mat * F;
                W_next -= R_F * W.col(j); // equivalent to daxpy.

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

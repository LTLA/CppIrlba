#ifndef IRLBA_LANCZOS_HPP
#define IRLBA_LANCZOS_HPP

#include "Eigen/Dense"
#include "utils.hpp"
#include <cmath>
#include <limits>

namespace irlba {

/**
 * @brief Perform Lanczos bidiagonalization on an input matrix.
 */
class LanczosBidiagonalization {
public:
    /**
     * Set the tolerance to use to define invariant subspaces.
     * 
     * @param e Tolerance, a positive number.
     *
     * @return A reference to the `LanczosBidiagonalization` instance.
     */
    LanczosBidiagonalization& set_eps(double e) {
        eps = e;
        return *this;
    }

    /**
     * Set the default tolerance for defining invariant subspaces.
     * This uses the same value as the **irlba** R package, i.e., the machine epsilon to the power of 0.8.
     * 
     * @return A reference to the `LanczosBidiagonalization` instance.
     */
    LanczosBidiagonalization& set_eps() {
        eps = default_eps;
        return *this;
    }

public:
    /**
     * Perform the Lanczos bidiagonalization on an input matrix, optionally with scaling and centering.
     * This implements Algorithm 2.1 described by Baglama and Reichel (2005).
     * Support is provided for centering and scaling without modifying `mat`.
     * Protection against invariant subspaces is also implemented.
     *
     * @tparam M Matrix class, most typically from the **Eigen** library.
     * We expect:
     * - A `rows()` method that returns the number of rows.
     * - A `cols()` method that returns the number of columns.
     * - A `*` method for matrix-vector multiplication.
     *   This should accept an `Eigen::VectorXd` of length equal to the number of columns as the right-hand argument,
     *   and return an `Eigen::VectorXd`-coercible object of length equal to the number of rows.
     * - An `adjoint()` method that returns an instance of any class that has a `*` method for matrix-vector multiplication.
     *   The method should accept an `Eigen::VectorXd` of length equal to the number of rows and return and return an `Eigen::VectorXd`-coercible object of length equal to the number of columns.
     * @tparam CENTER Either `Eigen::VectorXd` or `bool`.
     * @tparam CENTER Either `Eigen::VectorXd` or `bool`.
     * @tparam Engine A functor that, when called with no arguments, returns a random integer from a discrete uniform distribution.
     *
     * @param mat Input matrix.
     * @param center A vector of length equal to the number of columns of `mat`.
     * Each value is to be subtracted from the corresponding column of `mat`.
     * Alternatively `false`, if no centering is to be performed.
     * @param scale A vector of length equal to the number of columns of `mat`.
     * Each value should be positive and is used to divide the corresponding column of `mat`.
     * @param eng An instance of a random number `Engine`.
     * @param W Output matrix with number of rows equal to `mat.rows()`.
     * The size of the working subspace is defined from the number of columns.
     * The first `start` columns should contain orthonormal column vectors with non-zero L2 norms.
     * @param V Matrix with number of rows equal to `mat.cols()` and number of columns equal to `W.cols()`.
     * The first `start + 1` columns should contain orthonormal column vectors with non-zero L2 norms.
     * @param B Square matrix with number of rows and columns equal to the size of the working subspace.
     * Number of values is defined by `set_number()`.
     * @param start The dimension from which to start the bidiagonalization.
     *
     * @return
     * `W` is filled with orthonormal vectors, as is `V`.
     * `B` is filled with upper diagonal entries.
     */
    template<class M, class CENTER, class SCALE, class Engine>
    void run(const M& mat, Eigen::MatrixXd& W, Eigen::MatrixXd& V, Eigen::MatrixXd& B, const CENTER& center, const SCALE& scale, Engine& eng, int start = 0) {
        constexpr bool do_center = !std::is_same<CENTER, bool>::value;
        constexpr bool do_scale = !std::is_same<SCALE, bool>::value;

        int work = W.cols();
        orthog.set_size(work);
        F.resize(mat.cols());
        W_next.resize(mat.rows());

        // We assume that the starting column is already normalized, see argument description for 'V'.
        F = V.col(start);
        update_W_next(mat, center, scale, W, start);

        double S = W_next.norm();
        if (S < eps) {
            throw -4;
        }
        W_next /= S;
        W.col(start) = W_next;

        // The Lanczos iterations themselves.
        for (int j = start; j < work; ++j) {
            F.noalias() = mat.adjoint() * W.col(j);

            // Centering and scaling, if requested.
            if constexpr(do_center) {
                double beta = W.col(j).sum();
                F -= beta * center;
            }
            if constexpr(do_scale) {
                F = F.cwiseQuotient(scale);
            }

            F -= S * V.col(j); // equivalent to daxpy.
            orthog.run(V, F, j + 1);

            if (j + 1 < work) {
                double R_F = F.norm();

                if (R_F < eps) {
                    fill_with_random_normals(F, eng);
                    orthog.run(V, F, j + 1);
                    R_F = F.norm();
                    F /= R_F;
                    R_F = 0;
                } else {
                    F /= R_F;
                }

                V.col(j + 1) = F;

                B(j, j) = S;
                B(j, j + 1) = R_F;

                update_W_next(mat, center, scale, W, j+1);

                S = W_next.norm();
                if (S < eps) {
                    fill_with_random_normals(F, eng);
                    orthog.run(W, W_next, j + 1);
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
    /* This function just updates W_next based on the current F, i.e., the
     * initialization vector (when ncols = 0) or the residual vector
     * (otherwise). It will also mutate F if centering or scaling is requested.
     * 
     * It's worth noting that, when ncols > 0, 'F' at this point is equivalent
     * to 'x', the temporary buffer used in irlb.c's inner loop. 'F's original
     * value will not be used in the rest of the loop, so it's safe to mutate
     * 'F' inside the function here.
     */
    template<class M, class CENTER, class SCALE>
    void update_W_next(const M& mat, const CENTER& center, const SCALE& scale, const Eigen::MatrixXd& W, int ncols) {
        constexpr bool do_center = !std::is_same<CENTER, bool>::value;
        constexpr bool do_scale = !std::is_same<SCALE, bool>::value;

        // Applying the scaling.
        if constexpr(do_scale) {
            F = F.cwiseQuotient(scale);
        }

        W_next.noalias() = mat * F;

        // Applying the centering.
        if constexpr(do_center) {
            double beta = F.dot(center);
            for (auto& w : W_next) {
                w -= beta;
            }
        }

        // Full re-orthogonalization, using the left-most 'ncol' columns of W.
        // Recall that W_next will be the 'ncol + 1' column, i.e., W.col(ncol) in
        // 0-indexed terms, so we want to orthogonalize to all previous columns.
        if (ncols) {
            orthog.run(W, W_next, ncols);
        }
        return;
    }
public:
    /**
     * Obtain the residual vector, see algorithm 2.1 of Baglama and Reichel (2005).
     *
     * @return Vector of residuals of length equal to the number of columns of `mat` in `run()`.
     */
    const Eigen::VectorXd& residuals() const {
        return F;
    }
    
private:
    OrthogonalizeVector orthog;
    const double default_eps = std::pow(std::numeric_limits<double>::epsilon(), 0.8); 
    double eps = default_eps;

    Eigen::VectorXd F; 
    Eigen::VectorXd W_next;
};

}

#endif

#ifndef IRLBA_LANCZOS_HPP
#define IRLBA_LANCZOS_HPP

#include "Eigen/Dense"
#include "utils.hpp"
#include <cmath>
#include <limits>

namespace irlba {

/**
 * @brief Perform Lanczos bidiagonalization on an input matrix
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
     * Set the default tolerance to use to define invariant subspaces.
     * This inherits the same default as that defined from the **irlba** R package,
     * i.e., the machine epsilon to the power of 0.8.
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
     *
     * @tparam M Matrix class that supports `cols()`, `rows()`, `*` and `adjoint()`.
     * This is most typically a class from the Eigen library.
     * @tparam CENTER Either `Eigen::VectorXd` or `bool`.
     * @tparam CENTER Either `Eigen::VectorXd` or `bool`.
     * @tparam NORMSAMP A functor that, when called with no arguments, returns a random Normal value.
     *
     * @param mat Input matrix.
     * @param center A vector of length equal to the number of columns of `mat`.
     * Each value is to be subtracted from the corresponding column of `mat`.
     * Alternatively `false`, if no centering is to be performed.
     * @param scale A vector of length equal to the number of columns of `mat`.
     * Each value should be positive and is used to divide the corresponding column of `mat`.
     * @param norm An instance of a functor to generate normally distributed values.
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
    template<class M, class CENTER, class SCALE, class NORMSAMP>
    void run(const M& mat, Eigen::MatrixXd& W, Eigen::MatrixXd& V, Eigen::MatrixXd& B, const CENTER& center, const SCALE& scale, NORMSAMP& norm, int start = 0) {
        constexpr bool do_center = !std::is_same<CENTER, bool>::value;
        constexpr bool do_scale = !std::is_same<SCALE, bool>::value;

        int work = W.cols();
        orthog.set_size(work);
        F.resize(mat.cols());
        W_next.resize(mat.rows());

        // We assume that the starting column is already normalized.
        F = V.col(start);

        if constexpr(do_scale) {
            F = F.cwiseQuotient(scale);
        }

        W_next.noalias() = mat * F;

        if constexpr(do_center) {
            double beta = F.dot(center);
            for (auto& w : W_next) {
                w -= beta;
            }
        }

        if (start) {
            orthog.run(W, W_next, start);
        }

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
                    for (auto& f : F) { f = norm(); }
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

                // Re-using 'F' as 'x', the temporary buffer used in irlb.c's
                // inner loop. 'F's original value will not be used in the
                // rest of the loop, so no harm, no foul.
                auto& x = F;

                // Applying the scaling.
                if constexpr(do_scale) {
                    x = x.cwiseQuotient(scale);
                }

                W_next.noalias() = mat * x;

                // Applying the centering.
                if constexpr(do_center) {
                    double beta = x.dot(center);
                    for (auto& x : W_next) {
                        x -= beta;
                    }
                }

                // One round of classical Gram-Schmidt. 
                W_next -= R_F * W.col(j);

                // Full re-orthogonalization of W_{j+1}.
                orthog.run(W, W_next, j + 1);

                S = W_next.norm();
                if (S < eps) {
                    for (auto& w : W_next) { w = norm(); }
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

public:
    const Eigen::VectorXd& finalF() const {
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

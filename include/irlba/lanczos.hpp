#ifndef IRLBA_LANCZOS_HPP
#define IRLBA_LANCZOS_HPP

#include "Eigen/Dense"
#include "utils.hpp"
#include "wrappers.hpp"
#include "Options.hpp"
#include <cmath>
#include <limits>

namespace irlba {

namespace internal {

template<class EigenMatrix_, class EigenVector_>
void orthogonalize_vector(const EigenMatrix_& mat, EigenVector_& vec, size_t ncols, EigenVector_& tmp) {
    tmp.head(ncols).noalias() = mat.leftCols(ncols).adjoint() * vec;
    vec.noalias() -= mat.leftCols(ncols) * tmp.head(ncols);
}

template<class EigenVector_, class Matrix_>
struct LanczosWorkspace {
    LanczosWorkspace(const Matrix_& mat) : 
        F(mat.cols()), 
        W_next(mat.rows()), 
        orthog_tmp(mat.cols()), 
        work(wrapped_workspace(mat)),
        awork(wrapped_adjoint_workspace(mat))
    {}

public:
    EigenVector_ F; 
    EigenVector_ W_next;
    EigenVector_ orthog_tmp;
    WrappedWorkspace<Matrix_> work;
    WrappedAdjointWorkspace<Matrix_> awork;
};

/*
 * W is a matrix with number of rows equal to `mat.rows()`.
 * The size of the working subspace is defined as the number of columns.
 * The first `start` columns should contain orthonormal column vectors with non-zero L2 norms.
 * On output, the rest of `W` is filled with orthonormal vectors.
 *
 * V is a matrix with number of rows equal to `mat.cols()` and number of columns equal to `W.cols()`.
 * The first `start + 1` columns should contain orthonormal column vectors with non-zero L2 norms.
 * On output, the rest of `V` is filled with orthonormal vectors.
 *
 * B is a square matrix with number of rows and columns equal to the size of the working subspace.
 * On output, B is filled with upper diagonal entries, starting from the `start`-th row/column.
 */
template<class Matrix_, class EigenMatrix_, class EigenVector_, class Engine_>
void run_lanczos_bidiagonalization(
    const Matrix_& mat, 
    EigenMatrix_& W, 
    EigenMatrix_& V, 
    EigenMatrix_& B, 
    Engine_& eng, 
    LanczosWorkspace<EigenVector_, Matrix_>& inter, 
    Eigen::Index start, 
    const Options& options) 
{
    typedef typename EigenMatrix_::Scalar Float;
    Float raw_eps = options.invariant_subspace_tolerance;
    Float eps = (raw_eps < 0 ? std::pow(std::numeric_limits<Float>::epsilon(), 0.8) : raw_eps);

    Eigen::Index work = W.cols();
    auto& F = inter.F;
    auto& W_next = inter.W_next;
    auto& otmp = inter.orthog_tmp;

    F = V.col(start);
    wrapped_multiply(mat, F, inter.work, W_next); // i.e., W_next = mat * F;

    // If start = 0, there's nothing to orthogonalize against.
    if (start) {
        orthogonalize_vector(W, W_next, start, otmp);
    }

    Float S = W_next.norm();
    if (S < eps) {
        throw std::runtime_error("starting vector near the null space of the input matrix");
    }
    W_next /= S;
    W.col(start) = W_next;

    // The Lanczos iterations themselves, see algorithm 2.1 of Baglama and Reichel.
    for (Eigen::Index j = start; j < work; ++j) {
        wrapped_adjoint_multiply(mat, W.col(j), inter.awork, F); // i.e., F = mat.adjoint() * W.col(j);

        F -= S * V.col(j); // equivalent to daxpy.
        orthogonalize_vector(V, F, j + 1, otmp);

        if (j + 1 < work) {
            Float R_F = F.norm();

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

            wrapped_multiply(mat, F, inter.work, W_next); // i.e., W_next = mat * F;
            W_next -= R_F * W.col(j); // equivalent to daxpy.

            // Full re-orthogonalization, using the left-most 'j + 1' columns of W.
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
            // The paper's algorithm sets B(j + 1, j + 1) = S in the
            // j + 1 < work clause. But the irlba R package's C code 
            // sets it on the last loop, which gives the same result;
            // so to avoid any headaches, we do the same thing too.
            B(j, j) = S;
        }
    }
}

}

}

#endif

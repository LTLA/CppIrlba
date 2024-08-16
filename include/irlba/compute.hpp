#ifndef IRLBA_COMPUTE_HPP
#define IRLBA_COMPUTE_HPP

#include "Eigen/Dense"
#include "utils.hpp"
#include "lanczos.hpp"

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>

/**
 * @file compute.hpp
 * @brief Compute an approximate SVD with IRLBA.
 */

namespace irlba {

/**
 * @cond
 */
namespace internal {

template<class Matrix_, class EigenMatrix_, class EigenVector_>
void exact(const Matrix_& matrix, int requested_number, EigenMatrix_& outU, EigenMatrix_& outV, EigenVector_& outD) {
    Eigen::BDCSVD<EigenMatrix_> svd(matrix.rows(), matrix.cols(), Eigen::ComputeThinU | Eigen::ComputeThinV);

    if constexpr(internal::is_eigen<Matrix_>::value) {
        svd.compute(matrix);
    } else {
        auto adjusted = wrapped_realize<EigenMatrix_>(matrix);
        svd.compute(adjusted);
    }

    outD.resize(requested_number);
    outD = svd.singularValues().head(requested_number);

    outU.resize(matrix.rows(), requested_number);
    outU = svd.matrixU().leftCols(requested_number);

    outV.resize(matrix.cols(), requested_number);
    outV = svd.matrixV().leftCols(requested_number);

    return;
}

}
/**
 * @endcond
 */

/**
 * Implements the Implicitly Restarted Lanczos Bidiagonalization Algorithm (IRLBA) for fast truncated singular value decomposition.
 * This is heavily derived from the C code in the [**irlba** package](https://github.com/bwlewis/irlba),
 * with refactoring into C++ to use Eigen instead of LAPACK for much of the matrix algebra.
 *
 * @tparam Matrix_ Class satisfying the `MockMatrix` interface, or a floating-point `Eigen:Matrix` class.
 * @tparam EigenMatrix_ A floating-point `Eigen::Matrix` class.
 * @tparam EigenVector_ A floating-point `Eigen::Vector` class, typically of the same scalar type as `EigenMatrix_`.
 *
 * @param[in] matrix Input matrix.
 * Custom classes can also be used here to pass modified matrices that cannot be efficiently realized into the standard **Eigen** classes.
 * See the `wrappers.hpp` file for more details, along with the `Centered` and `Scaled` classes.
 * @param number Number of singular triplets to obtain.
 * @param[out] outU Output matrix where columns contain the first left singular vectors.
 * Dimensions are set automatically on output;
 * the number of columns is set to `number` and the number of rows is equal to the number of rows in `mat`.
 * @param[out] outV Output matrix where columns contain the first right singular vectors.
 * Dimensions are set automatically on output;
 * the number of columns is set to `number` and the number of rows is equal to the number of columns in `mat`.
 * @param[out] outD Vector to store the first singular values.
 * The length is set to `number` on output.
 * @param options Further options.
 *
 * @return A pair where the first entry indicates whether the algorithm converged,
 * and the second entry indicates the number of restart iterations performed.
 */
template<class Matrix_, class EigenMatrix_, class EigenVector_>
std::pair<bool, int> compute(const Matrix_& matrix, Eigen::Index number, EigenMatrix_& outU, EigenMatrix_& outV, EigenVector_& outD, const Options& options) {
    Eigen::Index smaller = std::min(matrix.rows(), matrix.cols());
    Eigen::Index requested_number = number;
    if (requested_number > smaller) {
        if (options.cap_number) {
            requested_number = smaller;
        } else {
            throw std::runtime_error("requested number of singular values cannot be greater than the smaller matrix dimension");
        }
    } else if (requested_number == smaller && !options.exact_for_large_number) {
        throw std::runtime_error("requested number of singular values must be less than the smaller matrix dimension for IRLBA iterations");
    }

    // Falling back to an exact SVD for small matrices or if the requested number is too large 
    // (not enough of a workspace). Hey, I don't make the rules.
    if ((options.exact_for_small_matrix && smaller < 6) || (options.exact_for_large_number && requested_number * 2 >= smaller)) {
        internal::exact(matrix, requested_number, outU, outV, outD);
        return std::make_pair(true, 0);
    }

    Eigen::Index work = std::min(requested_number + options.extra_work, smaller);

    EigenMatrix_ V(matrix.cols(), work);
    std::mt19937_64 eng(options.seed);
    if (options.initial) {
        auto init = reinterpret_cast<EigenVector_*>(options.initial);
        if (init->size() != V.rows()) {
            throw std::runtime_error("initialization vector does not have expected number of rows");
        }
        V.col(0) = *init;
    } else {
        internal::fill_with_random_normals(V, 0, eng);
    }
    V.col(0) /= V.col(0).norm();

    bool converged = false;
    int iter = 0;
    Eigen::Index k = 0;
    Eigen::JacobiSVD<EigenMatrix_> svd(work, work, Eigen::ComputeThinU | Eigen::ComputeThinV);

    internal::LanczosWorkspace<EigenVector_, Matrix_> lptmp(matrix);

    EigenMatrix_ W(matrix.rows(), work);
    EigenMatrix_ Wtmp(matrix.rows(), work);
    EigenMatrix_ Vtmp(matrix.cols(), work);

    EigenMatrix_ B(work, work);
    B.setZero(work, work);
    EigenVector_ res(work);
    EigenVector_ F(matrix.cols());

    EigenVector_ prevS(work);
    typename EigenMatrix_::Scalar svtol = options.singular_value_ratio_tolerance;
    typename EigenMatrix_::Scalar tol = options.convergence_tolerance;
    typename EigenMatrix_::Scalar svtol_actual = (svtol >= 0 ? svtol : tol);

    for (; iter < options.max_iterations; ++iter) {
        // Technically, this is only a 'true' Lanczos bidiagonalization
        // when k = 0. All other times, we're just recycling the machinery,
        // see the text below Equation 3.11 in Baglama and Reichel.
        internal::run_lanczos_bidiagonalization(matrix, W, V, B, eng, lptmp, k, options);

//            if (iter < 2) {
//                std::cout << "B is currently:\n" << B << std::endl;
//                std::cout << "W is currently:\n" << W << std::endl;
//                std::cout << "V is currently:\n" << V << std::endl;
//            }

        svd.compute(B);
        const auto& BS = svd.singularValues();
        const auto& BU = svd.matrixU();
        const auto& BV = svd.matrixV();

        // Checking for convergence.
        if (B(work - 1, work - 1) == 0) { // a.k.a. the final value of 'S' from the Lanczos iterations.
            converged = true;
            break;
        }

        const auto& F = lptmp.F; // i.e., the residuals, see Algorithm 2.1 of Baglama and Reichel.
        auto R_F = F.norm();

        // Computes the convergence criterion defined in on the LHS of
        // Equation 2.13 of Baglama and Riechel. (e_m is literally the unit
        // vector along the m-th dimension, so it ends up just being the
        // m-th row of the U matrix.) We expose it here as we will be
        // re-using the same values to update B, see below.
        res = R_F * BU.row(work - 1);

        Eigen::Index n_converged = 0;
        if (iter > 0) {
            auto Smax = *std::max_element(BS.begin(), BS.end());
            auto threshold = Smax * tol;

            for (Eigen::Index j = 0; j < work; ++j) {
                if (std::abs(res[j]) <= threshold) {
                    auto ratio = std::abs(BS[j] - prevS[j]) / BS[j];
                    if (ratio <= svtol_actual) {
                        ++n_converged;
                    }
                }
            }

            if (n_converged >= requested_number) {
                converged = true;
                break;
            }
        }
        prevS = BS;

        // Setting 'k'. This looks kinda weird, but this is deliberate,
        // see the text below Algorithm 3.1 of Baglama and Reichel.
        //
        // - Our n_converged is their k'.
        // - Our requested_number is their k.
        // - Our work is their m.
        // - Our k is their k + k''.
        if (n_converged + requested_number > k) {
            k = n_converged + requested_number;
        }
        if (k > work - 3) {
            k = work - 3;
        }
        if (k < 1) {
            k = 1;
        }

        Vtmp.leftCols(k).noalias() = V * BV.leftCols(k);
        V.leftCols(k) = Vtmp.leftCols(k);

        // See Equation 3.2 of Baglama and Reichel, where our 'V' is their
        // 'P', and our 'F / R_F' is their 'p_{m+1}' (Equation 2.2).  'F'
        // was orthogonal to the old 'V' and so it should still be
        // orthogonal to the new left-most columns of 'V'; the input
        // expectations of the Lanczos bidiagonalization are still met, so
        // V is ok to use in the next run_lanczos_bidiagonalization().
        V.col(k) = F / R_F; 

        Wtmp.leftCols(k).noalias() = W * BU.leftCols(k);
        W.leftCols(k) = Wtmp.leftCols(k);

        B.setZero(work, work);
        for (Eigen::Index l = 0; l < k; ++l) {
            B(l, l) = BS[l];

            // This assignment looks weird but is deliberate, see Equation
            // 3.6 of Baglama and Reichel. Happily, this is the same value
            // used to determine convergence in 2.13, so we just re-use it.
            // (See the equation just above Equation 3.5; I think they 
            // misplaced a tilde on the final 'u', given no other 'u' has
            // a B_m superscript as well as a tilde.)
            B(l, k) = res[l]; 
        }
    }

    // See Equation 2.11 of Baglama and Reichel for how to get from B's
    // singular triplets to mat's singular triplets.
    outD.resize(requested_number);
    outD = svd.singularValues().head(requested_number);

    outU.resize(matrix.rows(), requested_number);
    outU.noalias() = W * svd.matrixU().leftCols(requested_number);

    outV.resize(matrix.cols(), requested_number);
    outV.noalias() = V * svd.matrixV().leftCols(requested_number);

    return std::make_pair(converged, iter + 1);
}

/**
 * @brief Result of the IRLBA-based decomposition.
 * @tparam EigenMatrix_ A floating-point `Eigen::Matrix` class.
 * @tparam EigenVector_ A floating-point `Eigen::Vector` class, typically of the same scalar type as `EigenMatrix_`.
 */
template<class EigenMatrix_, class EigenVector_>
struct Results {
    /**
     * The left singular vectors, stored as columns of `U`.
     * The number of rows in `U` is equal to the number of rows in the input matrix,
     * and the number of columns is equal to the number of requested vectors.
     */
    EigenMatrix_ U;

    /**
     * The right singular vectors, stored as columns of `U`.
     * The number of rows in `U` is equal to the number of columns in the input matrix,
     * and the number of columns is equal to the number of requested vectors.
     */
    EigenMatrix_ V;

    /**
     * The requested number of singular values, ordered by decreasing value.
     * These correspond to the vectors in `U` and `V`.
     */
    EigenVector_ D;

    /**
     * The number of restart iterations performed.
     */
    int iterations;

    /**
     * Whether the algorithm converged.
     */
    bool converged;
};

/** 
 * Convenient overload of `compute()` that allocates memory for the output matrices of the SVD.
 *
 * @tparam EigenMatrix_ A floating-point `Eigen::Matrix` class.
 * @tparam EigenVector_ A floating-point `Eigen::Vector` class, typically of the same scalar type as `EigenMatrix_`.
 * @tparam Matrix_ Class satisfying the `MockMatrix` interface, or a floating-point `Eigen:Matrix` class.
 *
 * @param[in] matrix Input matrix.
 * @param number Number of singular triplets to obtain.
 * @param options Further options.
 *
 * @return A `Results` object containing the singular vectors and values, as well as some statistics on convergence.
 */
template<class EigenMatrix_ = Eigen::MatrixXd, class EigenVector_ = Eigen::VectorXd, class Matrix_>
Results<EigenMatrix_, EigenVector_> compute(const Matrix_& matrix, Eigen::Index number, const Options& options) {
    Results<EigenMatrix_, EigenVector_> output;
    auto stats = compute(matrix, number, output.U, output.V, output.D, options);
    output.converged = stats.first;
    output.iterations = stats.second;
    return output;
}

/**
 * Convenient overload of `compute()` that handles the centering and scaling.
 *
 * @tparam Matrix_ A floating-point `Eigen::Matrix` or equivalent sparse matrix class.
 * @tparam EigenMatrix_ A floating-point `Eigen::Matrix` class.
 * @tparam EigenVector_ A floating-point `Eigen::Vector` class, typically of the same scalar type as `EigenMatrix_`.
 *
 * @param[in] matrix Input matrix.
 * @param center Should the matrix be centered by column?
 * @param scale Should the matrix be scaled to unit variance for each column?
 * @param number Number of singular triplets to obtain.
 * @param[out] outU Output matrix where columns contain the first left singular vectors.
 * Dimensions are set automatically on output;
 * the number of columns is set to `number` and the number of rows is equal to the number of rows in `mat`.
 * @param[out] outV Output matrix where columns contain the first right singular vectors.
 * Dimensions are set automatically on output;
 * the number of columns is set to `number` and the number of rows is equal to the number of columns in `mat`.
 * @param[out] outD Vector to store the first singular values.
 * The length is set to `number` on output.
 * @param options Further options.
 *
 * @return A pair where the first entry indicates whether the algorithm converged,
 * and the second entry indicates the number of restart iterations performed.
 *
 * Centering is performed by subtracting each element of `center` from the corresponding column of `mat`.
 * Scaling is performed by dividing each column of `mat` by the corresponding element of `scale` (after any centering has been applied).
 * Note that `scale=true` requires `center=true` to guarantee unit variance along each column. 
 * No scaling is performed when the variance of a column is zero, so as to avoid divide-by-zero errors. 
 */
template<class Matrix_, class EigenMatrix_, class EigenVector_>
std::pair<bool, int> compute(const Matrix_& matrix, bool center, bool scale, Eigen::Index number, EigenMatrix_& outU, EigenMatrix_& outV, EigenVector_& outD, const Options& options) {
    if (!scale && !center) {
        return compute(matrix, number, outU, outV, outD, options);
    }

    auto nr = matrix.rows();
    auto nc = matrix.cols();

    Eigen::VectorXd center0;
    if (center) {
        if (nr < 1) {
            throw std::runtime_error("cannot center with no observations");    
        }
        center0.resize(nc);
    }

    Eigen::VectorXd scale0;
    if (scale) {
        if (nr < 2) {
            throw std::runtime_error("cannot scale with fewer than two observations");    
        }
        scale0.resize(nc);
    }

    for (Eigen::Index i = 0; i < nc; ++i) {
        double mean = 0;
        if (center) {
            mean = matrix.col(i).sum() / nr;
            center0[i] = mean;
        }
        if (scale) {
            EigenVector_ current = matrix.col(i); // force it to be a Vector, even if it's a sparse matrix.
            typename EigenMatrix_::Scalar var = 0;
            for (auto x : current) {
                var += (x - mean)*(x - mean);
            }

            if (var) {
                scale0[i] = std::sqrt(var/(nr - 1));
            } else {
                scale0[i] = 1;
            }
        }
    }

    if (center) {
        Centered<EigenMatrix_, EigenVector_> centered(matrix, center0);
        if (scale) {
            auto centered_scaled = make_Scaled<true>(centered, scale0, true);
            return compute(centered_scaled, number, outU, outV, outD, options);
        } else {
            return compute(centered, number, outU, outV, outD, options);
        }
    } else {
        auto scaled = make_Scaled<true>(matrix, scale0, true);
        return compute(scaled, number, outU, outV, outD, options);
    }
}

/** 
 * Convenient overload of `compute()` with centering and scaling, which allocates memory for the output matrices of the SVD.
 *
 * @tparam Matrix_ A floating-point `Eigen::Matrix` or equivalent sparse matrix class.
 * @tparam EigenMatrix_ A floating-point `Eigen::Matrix` class.
 * @tparam EigenVector_ A floating-point `Eigen::Vector` class, typically of the same scalar type as `EigenMatrix_`.
 *
 * @param[in] matrix Input matrix.
 * @param center Should the matrix be centered by column?
 * @param scale Should the matrix be scaled to unit variance for each column?
 * @param number Number of singular triplets to obtain.
 * @param options Further options.
 *
 * @return A `Results` object containing the singular vectors and values, as well as some statistics on convergence.
 */
template<class EigenMatrix_ = Eigen::MatrixXd, class EigenVector_ = Eigen::VectorXd, class Matrix_>
Results<EigenMatrix_, EigenVector_> compute(const Matrix_& matrix, bool center, bool scale, Eigen::Index number, const Options& options) {
    Results<EigenMatrix_, EigenVector_> output;
    auto stats = compute(matrix, center, scale, number, output.U, output.V, output.D, options);
    output.converged = stats.first;
    output.iterations = stats.second;
    return output;
}

}

#endif

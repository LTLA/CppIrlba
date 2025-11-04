#ifndef IRLBA_COMPUTE_HPP
#define IRLBA_COMPUTE_HPP

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <type_traits>

#include "utils.hpp"
#include "lanczos.hpp"
#include "Matrix/simple.hpp"

#include "sanisizer/sanisizer.hpp"
#include "Eigen/Dense"

/**
 * @file compute.hpp
 * @brief Compute an approximate SVD with IRLBA.
 */

namespace irlba {

/**
 * @cond
 */
template<typename EigenMatrix_>
using JacobiSVD = Eigen::JacobiSVD<EigenMatrix_, Eigen::ComputeThinU | Eigen::ComputeThinV>;

template<class Matrix_, class EigenMatrix_, class EigenVector_>
void exact(const Matrix_& matrix, const Eigen::Index requested_number, EigenMatrix_& outU, EigenMatrix_& outV, EigenVector_& outD) {
    JacobiSVD<EigenMatrix_> svd(matrix.rows(), matrix.cols());

    auto realizer = matrix.new_known_realize_workspace();
    EigenMatrix_ buffer;
    svd.compute(realizer->realize(buffer));

    outD.resize(requested_number);
    outD = svd.singularValues().head(requested_number);

    outU.resize(matrix.rows(), requested_number);
    outU = svd.matrixU().leftCols(requested_number);

    outV.resize(matrix.cols(), requested_number);
    outV = svd.matrixV().leftCols(requested_number);
}

// Basically (requested * 2 >= smaller), but avoiding overflow from the product.
inline bool requested_greater_than_or_equal_to_half_smaller(const Eigen::Index requested, const Eigen::Index smaller) {
    const Eigen::Index half_smaller = smaller / 2;
    if (requested == half_smaller) {
        return smaller % 2 == 0;
    } else {
        return requested > half_smaller;
    }
}

// Basically min(requested_number + extra_work, smaller), but avoiding overflow from the sum.
inline Eigen::Index choose_requested_plus_extra_work_or_smaller(const Eigen::Index requested_number, const int extra_work, const Eigen::Index smaller) {
    if (requested_number >= smaller) {
        return smaller;
    } else {
        // This is guaranteed to fit into an Eigen::Index;
        // either it's equal to 'smaller' or it's less than 'smaller'.
        return requested_number + sanisizer::min(smaller - requested_number, extra_work);
    }
}

// Setting 'k'. This looks kinda weird, but this is deliberate, see the text below Algorithm 3.1 of Baglama and Reichel.
//
// - Our n_converged is their k'.
// - Our requested_number is their k.
// - Our work is their m.
// - Our returned k is their k + k''.
//
// They don't mention anything about the value corresponding to our input k, but I guess we always want k to increase.
// So, if our proposed new value of k is lower than our current value of k, we just pick the latter.
inline Eigen::Index update_k(Eigen::Index k, const Eigen::Index requested_number, const Eigen::Index n_converged, const Eigen::Index work) {
    // Here, the goal is to reproduce this code from irlba's convtests() function, but without any risk of wraparound or overflow.
    //
    // if (k < requested_number + n_converged) {
    //    k = requested_number + n_converged;
    // }
    // if (k > work - 3) {
    //    k = work - 3;
    // }
    // if (k < 1) {
    //    k = 1;
    // }

    if (work <= 3) {
        return 1;
    }
    const Eigen::Index limit = work - 3;

    const auto less_than_requested_plus_converged = [&](const Eigen::Index val) -> bool {
        return val < requested_number || static_cast<Eigen::Index>(val - requested_number) < n_converged;
    };

    if (less_than_requested_plus_converged(k)) {
        if (less_than_requested_plus_converged(limit)) {
            return limit;
        } else {
            const Eigen::Index output = requested_number + n_converged;
            return std::max(output, static_cast<Eigen::Index>(1));
        }
    } else {
        const Eigen::Index output = std::min(k, limit);
        return std::max(output, static_cast<Eigen::Index>(1));
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
 * @tparam Matrix_ Class satisfying the `Matrix<EigenVector_, EigenMatrix_>` interface. 
 * @tparam EigenMatrix_ A dense floating-point `Eigen::Matrix` class to store the output.
 * @tparam EigenVector_ A floating-point `Eigen::Vector` class to store the output, typically of the same scalar type as `EigenMatrix_`.
 *
 * @param[in] matrix Input matrix.
 * @param number Number of singular triplets to obtain.
 * The returned number of triplets may be lower, see `Options::cap_number` for details.
 * @param[out] outU Output matrix containing the left singular vectors corresponding to the largest singular values.
 * Each column corresponds to a left singular vector, of which there are `number` (or less, depending on `Options::cap_number`).
 * The number of rows is equal to the number of rows in `mat`.
 * @param[out] outV Output matrix containing the right singular vectors corresponding to the largest singular values.
 * Each column corresponds to a right singular vector, of which there are `number` (or less, depending on `Options::cap_number`).
 * The number of rows is equal to the number of columns in `mat`.
 * @param[out] outD Output vector containing the largest singular values, ordered by decreasing size.
 * This has length equal to `number` (or lower, depending on `Options::cap_number`).
 * @param options Further options.
 *
 * @return A pair where the first entry indicates whether the algorithm converged,
 * and the second entry indicates the number of restart iterations performed.
 */
template<class Matrix_, class EigenMatrix_, class EigenVector_>
std::pair<bool, int> compute(
    const Matrix_& matrix,
    const Eigen::Index number,
    EigenMatrix_& outU,
    EigenMatrix_& outV,
    EigenVector_& outD,
    const Options<EigenVector_>& options
) {
    const Eigen::Index smaller = std::min(matrix.rows(), matrix.cols());
    Eigen::Index requested_number = sanisizer::cast<Eigen::Index>(number);
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
    if (
        (options.exact_for_small_matrix && smaller < 6) ||
        (options.exact_for_large_number && requested_greater_than_or_equal_to_half_smaller(requested_number, smaller))
    ) {
        exact(matrix, requested_number, outU, outV, outD);
        return std::make_pair(true, 0);
    }

    Eigen::Index work = choose_requested_plus_extra_work_or_smaller(requested_number, options.extra_work, smaller);
    if (work == 0) {
        throw std::runtime_error("number of requested dimensions must be positive");
    }

    // Don't worry about sanitizing dimensions for Eigen constructors,
    // as the former are stored as Eigen::Index and the latter accepts Eigen::Index inputs.
    EigenMatrix_ V(matrix.cols(), work);
    std::mt19937_64 eng(options.seed);
    if (options.initial.has_value()) {
        const auto& init = *(options.initial);
        if (init.size() != V.rows()) {
            throw std::runtime_error("initialization vector does not have expected number of rows");
        }
        V.col(0) = init;
    } else {
        fill_with_random_normals(V, 0, eng);
    }
    V.col(0) /= V.col(0).norm();

    bool converged = false;
    int iter = 0;
    Eigen::Index k = 0;
    JacobiSVD<EigenMatrix_> svd(work, work);

    LanczosWorkspace<EigenVector_, Matrix_> lpwork(matrix);

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
        run_lanczos_bidiagonalization(lpwork, W, V, B, eng, k, options);

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

        const auto& F = lpwork.F; // i.e., the residuals, see Algorithm 2.1 of Baglama and Reichel.
        auto R_F = F.norm();

        // Computes the convergence criterion defined in on the LHS of
        // Equation 2.13 of Baglama and Riechel. (e_m is literally the unit
        // vector along the m-th dimension, so it ends up just being the
        // m-th row of the U matrix.) We expose it here as we will be
        // re-using the same values to update B, see below.
        res = R_F * BU.row(work - 1);

        Eigen::Index n_converged = 0;
        if (iter > 0) {
            const auto Smax = *std::max_element(BS.begin(), BS.end());
            const auto threshold = Smax * tol;

            for (Eigen::Index j = 0; j < work; ++j) {
                if (std::abs(res[j]) <= threshold) {
                    const auto ratio = std::abs(BS[j] - prevS[j]) / BS[j];
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

        k = update_k(k, requested_number, n_converged, work);
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
        for (I<decltype(k)> l = 0; l < k; ++l) {
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

    return std::make_pair(converged, (converged ? iter + 1 : iter));
}

/**
 * Convenient overload of `compute()` that accepts an **Eigen** matrix directly.
 *
 * @tparam InputEigenMatrix_ An **Eigen** matrix class containing the input data.
 * @tparam OutputEigenMatrix_ A dense floating-point `Eigen::Matrix` class to store the output.
 * @tparam EigenVector_ A floating-point `Eigen::Vector` class to store the output, typically of the same scalar type as `EigenMatrix_`.
 *
 * @param[in] matrix Input matrix.
 * @param number Number of singular triplets to obtain.
 * The returned number of triplets may be lower, see `Options::cap_number` for details.
 * @param[out] outU Output matrix containing the left singular vectors corresponding to the largest singular values.
 * Each column corresponds to a left singular vector, of which there are `number` (or less, depending on `Options::cap_number`).
 * The number of rows is equal to the number of rows in `mat`.
 * @param[out] outV Output matrix containing the right singular vectors corresponding to the largest singular values.
 * Each column corresponds to a right singular vector, of which there are `number` (or less, depending on `Options::cap_number`).
 * The number of rows is equal to the number of columns in `mat`.
 * @param[out] outD Output vector containing the largest singular values, ordered by decreasing size.
 * This has length equal to `number` (or lower, depending on `Options::cap_number`).
 * @param options Further options.
 *
 * @return A pair where the first entry indicates whether the algorithm converged,
 * and the second entry indicates the number of restart iterations performed.
 */
template<class InputEigenMatrix_, class OutputEigenMatrix_, class EigenVector_>
std::pair<bool, int> compute_simple(
    const InputEigenMatrix_& matrix,
    Eigen::Index number,
    OutputEigenMatrix_& outU,
    OutputEigenMatrix_& outV,
    EigenVector_& outD,
    const Options<EigenVector_>& options
) {
    SimpleMatrix<EigenVector_, OutputEigenMatrix_, const InputEigenMatrix_*> wrapped(&matrix);

    // Force the compiler to use virtual dispatch to avoid realizing multiple
    // instances of this function for each InputEigenMatrix_ type. If you
    // don't like this, call compute() directly.
    typedef Matrix<EigenVector_, OutputEigenMatrix_> Interface; 

    return compute<Interface>(wrapped, number, outU, outV, outD, options);
}

/**
 * @brief Results of the IRLBA-based SVD by `compute()`.
 * @tparam EigenMatrix_ A dense floating-point `Eigen::Matrix` class.
 * @tparam EigenVector_ A floating-point `Eigen::Vector` class, typically of the same scalar type as `EigenMatrix_`.
 */
template<class EigenMatrix_, class EigenVector_>
struct Results {
    /**
     * Matrix of left singular vectors corresponding to the largest singular values.
     * Each column corresponds to a left singular vector, the number of which is no greater than `number`.
     * The number of rows is equal to the number of rows in the input matrix.
     */
    EigenMatrix_ U;

    /**
     * Matrix of right singular vectors corresponding to the largest singular values.
     * Each column corresponds to a right singular vector, the number of which is no greater than `number`.
     * The number of rows is equal to the number of columns in the input matrix.
     */
    EigenMatrix_ V;

    /**
     * Vector of the largest singular values, ordered by decreasing value.
     * The number of values is no greater than `number`.
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
 * @tparam EigenMatrix_ A dense floating-point `Eigen::Matrix` class to store the output.
 * @tparam EigenVector_ A floating-point `Eigen::Vector` class for the output, typically of the same scalar type as `EigenMatrix_`.
 * @tparam Matrix_ Class satisfying the `Matrix<EigenVector_, EigenMatrix_>` interface. 
 *
 * @param[in] matrix Input matrix.
 * @param number Number of singular triplets to obtain.
 * @param options Further options.
 *
 * @return A `Results` object containing the singular vectors and values, as well as some statistics on convergence.
 */
template<class EigenMatrix_ = Eigen::MatrixXd, class EigenVector_ = Eigen::VectorXd, class Matrix_>
Results<EigenMatrix_, EigenVector_> compute(const Matrix_& matrix, Eigen::Index number, const Options<EigenVector_>& options) {
    Results<EigenMatrix_, EigenVector_> output;
    const auto stats = compute(matrix, number, output.U, output.V, output.D, options);
    output.converged = stats.first;
    output.iterations = stats.second;
    return output;
}

/** 
 * Convenient overload of `compute()` that accepts an **Eigen** matrix directly.
 *
 * @tparam InputEigenMatrix_ An **Eigen** matrix class containing the input data.
 * @tparam OutputEigenMatrix_ A dense floating-point `Eigen::Matrix` class for the output.
 * @tparam EigenVector_ A floating-point `Eigen::Vector` class for the output, typically of the same scalar type as `OutputEigenMatrix_`.
 * @tparam Matrix_ Class satisfying the `Matrix<EigenVector_, EigenMatrix_>` interface. 
 *
 * @param[in] matrix Input matrix.
 * @param number Number of singular triplets to obtain.
 * @param options Further options.
 *
 * @return A `Results` object containing the singular vectors and values, as well as some statistics on convergence.
 */
template<class OutputEigenMatrix_ = Eigen::MatrixXd, class EigenVector_ = Eigen::VectorXd, class InputEigenMatrix_>
Results<OutputEigenMatrix_, EigenVector_> compute_simple(const InputEigenMatrix_& matrix, Eigen::Index number, const Options<EigenVector_>& options) {
    Results<OutputEigenMatrix_, EigenVector_> output;
    const auto stats = compute_simple(matrix, number, output.U, output.V, output.D, options);
    output.converged = stats.first;
    output.iterations = stats.second;
    return output;
}

}

#endif

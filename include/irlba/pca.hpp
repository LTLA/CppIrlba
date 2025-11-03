#ifndef IRLBA_PCA_HPP
#define IRLBA_PCA_HPP

#include <cmath>
#include <utility>
#include <stdexcept>

#include "compute.hpp"
#include "Matrix/simple.hpp"
#include "Matrix/centered.hpp"
#include "Matrix/scaled.hpp"

#include "Eigen/Dense"

/**
 * @file pca.hpp
 * @brief Perform PCA with IRLBA.
 */

namespace irlba {

/**
 * Perform a principal components analysis (PCA) using IRLBA for the underlying SVD.
 * This applies deferred centering and scaling via the `CenteredMatrix` and `ScaledMatrix` classes, respectively.
 *
 * @tparam InputEigenMatrix_ An **Eigen** matrix class containing the input data.
 * @tparam OutputEigenMatrix_ A dense floating-point `Eigen::Matrix` class for the output.
 * @tparam EigenVector_ A floating-point `Eigen::Vector` class for the output, typically of the same scalar type as `OutputEigenMatrix_`.
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
template<class InputEigenMatrix_, class OutputEigenMatrix_, class EigenVector_>
std::pair<bool, int> pca(
    const InputEigenMatrix_& matrix,
    bool center,
    bool scale,
    Eigen::Index number,
    OutputEigenMatrix_& outU,
    OutputEigenMatrix_& outV,
    EigenVector_& outD,
    const Options& options
) {
    // Reduce the size of the binary by forcing all compute() calls to use virtual dispatch,
    // rather than realizing four new instances of the same function with different Matrix subclasses.
    typedef Matrix<EigenVector_, OutputEigenMatrix_> Interface;
    SimpleMatrix<EigenVector_, OutputEigenMatrix_, I<decltype(&matrix)> > wrapped(&matrix);
    if (!scale && !center) {
        return compute<Interface>(wrapped, number, outU, outV, outD, options);
    }

    const Eigen::Index nr = matrix.rows();
    const Eigen::Index nc = matrix.cols();

    EigenVector_ center0;
    if (center) {
        if (nr < 1) {
            throw std::runtime_error("cannot center with no observations");    
        }
        center0.resize(nc);
    }

    EigenVector_ scale0;
    if (scale) {
        if (nr < 2) {
            throw std::runtime_error("cannot scale with fewer than two observations");    
        }
        scale0.resize(nc);
    }

    for (Eigen::Index i = 0; i < nc; ++i) {
        typename EigenVector_::Scalar mean = 0;
        if (center) {
            mean = matrix.col(i).sum() / nr;
            center0[i] = mean;
        }
        if (scale) {
            EigenVector_ current = matrix.col(i); // force it to be a Vector, even if it's a sparse matrix.
            typename EigenVector_::Scalar var = 0;
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
        CenteredMatrix<EigenVector_, OutputEigenMatrix_, I<decltype(&wrapped)>, I<decltype(&center0)> > centered(&wrapped, &center0);
        if (scale) {
            ScaledMatrix<EigenVector_, OutputEigenMatrix_, I<decltype(&centered)>, I<decltype(&scale0)> > centered_scaled(&centered, &scale0, true, true);
            return compute<Interface>(centered_scaled, number, outU, outV, outD, options);
        } else {
            return compute<Interface>(centered, number, outU, outV, outD, options);
        }
    } else {
        ScaledMatrix<EigenVector_, OutputEigenMatrix_, I<decltype(&wrapped)>, I<decltype(&scale0)> > scaled(&wrapped, &scale0, true, true);
        return compute<Interface>(scaled, number, outU, outV, outD, options);
    }
}

/** 
 * Convenient overload of `pca()` that allocates memory for the output matrices of the SVD.
 *
 * @tparam InputEigenMatrix_ An **Eigen** matrix class containing the input data.
 * @tparam OutputEigenMatrix_ A dense floating-point `Eigen::Matrix` class for the output.
 * @tparam EigenVector_ A floating-point `Eigen::Vector` class for the output, typically of the same scalar type as `EigenMatrix_`.
 *
 * @param[in] matrix Input matrix.
 * @param center Should the matrix be centered by column?
 * @param scale Should the matrix be scaled to unit variance for each column?
 * @param number Number of singular triplets to obtain.
 * @param options Further options.
 *
 * @return A `Results` object containing the singular vectors and values, as well as some statistics on convergence.
 */
template<class OutputEigenMatrix_ = Eigen::MatrixXd, class EigenVector_ = Eigen::VectorXd, class InputEigenMatrix_>
Results<OutputEigenMatrix_, EigenVector_> pca(const InputEigenMatrix_& matrix, bool center, bool scale, Eigen::Index number, const Options& options) {
    Results<OutputEigenMatrix_, EigenVector_> output;
    const auto stats = pca(matrix, center, scale, number, output.U, output.V, output.D, options);
    output.converged = stats.first;
    output.iterations = stats.second;
    return output;
}

}

#endif
